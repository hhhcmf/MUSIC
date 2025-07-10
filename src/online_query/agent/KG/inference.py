from online_query.agent.KG.KGCoordinator import KGCoordinator
from online_query.phase import PreprocessPhase, KnowledgeOrganizerPhase, AnswerGeneratePhase, AnswerValidatePhase
from utils.proprecss_llm_answer import extract_intermediate_answers
from collections import defaultdict
import time
import asyncio

class ReasoningPhase:
    def __init__(self, base_config, kg_reason_processor):
        self.base_config = base_config
        self.kg_reason_processor = kg_reason_processor
        self.preprocess_phase = PreprocessPhase(base_config)
        self.knowledge_organize_phase = KnowledgeOrganizerPhase(base_config)
        self.answer_generate_phase = AnswerGeneratePhase(base_config, kg_reason_processor.kg_config)
        self.answer_validate_phase = AnswerValidatePhase(base_config, kg_reason_processor.kg_config)
        self.KG_coordinator = KGCoordinator(base_config, kg_reason_processor)

    async def run(self, orginal_question, question, question_embedding, data_id_list, kge_flag, summarize_flag=0):
        question_entity_list = self.preprocess_phase.run(orginal_question, question)
        self.base_config.LLM.init_message()

        question_entities_information, retrieved_entity_data_list, _, _ = self.KG_coordinator.kg_reason_processor.get_subgraph_entity_information(
            question, data_id_list, question_entity_list, question_embedding, kge_flag
        )

        entities_information = ''
        table_records_data_list, image_qa_list, text_entity_data_list, data_inference_time_one = await self.KG_coordinator.get_agent_related_data(
            question, '', retrieved_entity_data_list, summarize_flag
        )

        agent_response_information, passage_information, image_qa_information, table_record_information = self.KG_coordinator.kg_reason_processor.get_agent_response_information(
            question, question_embedding, self.KG_coordinator.text_entity_data_list, self.KG_coordinator.image_qa_list, self.KG_coordinator.table_records_list
        )

        userful_information = self.knowledge_organize_phase.run(question, '', agent_response_information)

        answerable = False
        reasonable = False
        final_answer = ''
        answer_dict = defaultdict(int)
        supporting_id_list_dict = {}
        repeat_num = 0
        all_intermediate_answer_entity_data_list = []

        if self.base_config.args.dataset == 'webqa':
            short_flag = 0
        else:
            short_flag = 1

        while not answerable and not reasonable:
            answerable, temp_answer, supporting_id_list = self.answer_generate_phase.run(question, userful_information, short_flag)
            if answerable:
                final_answer = temp_answer
                answer_lower = temp_answer.lower()
                answer_dict[answer_lower] += 1
                supporting_id_list_dict[answer_lower] = supporting_id_list

                entity_information, agent_response_information, table_records_list, image_qa_list, text_entity_data_list = self.KG_coordinator.get_releted_entity_information(
                    userful_information, question, question_embedding, temp_answer, data_id_list
                )
                reasonable = self.answer_validate_phase.run(question, temp_answer, entity_information, agent_response_information)

                if reasonable:
                    return temp_answer, supporting_id_list

                answerable = False
                self.base_config.LLM.init_message()

                if repeat_num < 2 or not answer_dict:
                    intermediate_answer_entity_data_list, _ = await self.KG_coordinator.inference(
                        question, question_embedding, userful_information, text_entity_data_list, image_qa_list, table_records_data_list
                    )
                    intermediate_entity_set = set(intermediate_answer_entity_data_list)
                    all_intermediate_entity_set = set(all_intermediate_answer_entity_data_list)

                    if intermediate_entity_set.issubset(all_intermediate_entity_set):
                        repeat_num += 1
                    else:
                        all_intermediate_answer_entity_data_list.extend(intermediate_answer_entity_data_list)
                        all_intermediate_answer_entity_data_list = list(set(all_intermediate_answer_entity_data_list))
                else:
                    if answer_dict:
                        answer = max(answer_dict, key=lambda k: answer_dict[k])
                        supporting_id_list = supporting_id_list_dict[answer]
                        return answer, supporting_id_list

                entities_information = await self.KG_coordinator.get_related_information_from_entity_data_list(
                    question, self.KG_coordinator.retrieved_entity_data_list, question_embedding
                )
                agent_response_information, _, _, _ = self.KG_coordinator.kg_reason_processor.get_agent_response_information(
                    question, question_embedding, self.KG_coordinator.text_entity_data_list, self.KG_coordinator.image_qa_list, self.KG_coordinator.table_records_list
                )
                userful_information = self.knowledge_organize_phase.run(question, entities_information, agent_response_information)

            else:
                self.base_config.LLM.init_message()

                if repeat_num < 2 or not answer_dict:
                    intermediate_answer_entity_data_list, _ = await self.KG_coordinator.inference(
                        question, question_embedding, userful_information, text_entity_data_list, image_qa_list, table_records_data_list
                    )
                    intermediate_entity_set = set(intermediate_answer_entity_data_list)
                    all_intermediate_entity_set = set(all_intermediate_answer_entity_data_list)

                    if intermediate_entity_set.issubset(all_intermediate_entity_set):
                        repeat_num += 1
                    else:
                        all_intermediate_answer_entity_data_list.extend(intermediate_answer_entity_data_list)
                        all_intermediate_answer_entity_data_list = list(set(all_intermediate_answer_entity_data_list))
                else:
                    if answer_dict:
                        answer = max(answer_dict, key=lambda k: answer_dict[k])
                        supporting_id_list = supporting_id_list_dict[answer]
                        return answer, supporting_id_list

                entities_information = await self.KG_coordinator.get_related_information_from_entity_data_list(
                    question, self.KG_coordinator.retrieved_entity_data_list, question_embedding
                )
                agent_response_information, _, _, _ = self.KG_coordinator.kg_reason_processor.get_agent_response_information(
                    question, question_embedding, self.KG_coordinator.text_entity_data_list, self.KG_coordinator.image_qa_list, self.KG_coordinator.table_records_list
                )
                userful_information = self.knowledge_organize_phase.run(question, entities_information, agent_response_information)

        if answer_dict:
            answer = max(answer_dict, key=lambda k: answer_dict[k])
            supporting_id_list = supporting_id_list_dict[answer]
            return answer, supporting_id_list

        return final_answer, []

    