class OnlineQueryPipeline:
    def __init__(self, answer_judge, answer_generator, reasoning_engine, info_organizer, logger):
        self.answer_judge = answer_judge
        self.answer_generator = answer_generator
        self.reasoning_engine = reasoning_engine
        self.info_organizer = info_organizer
        self.logger = logger

    def query(self, question, question_embedding, kg_entities_name, db_path, img_folder, entity_link_hash):
        answerable = False
        information = {}
        
        while not answerable:
            # Step 1: 判断问题是否可回答
            answerable = self.answer_judge.is_answerable(question)

            if answerable:
                # Step 2: 生成初始答案
                answer = self.answer_generator.generate_answer(question, information)

                if 'no data' in answer.lower() or 'available' in answer.lower():
                    answerable = False
                    # Step 3: 执行初步推理
                    reasoning_result = self.reasoning_engine.perform_reasoning(
                        question, question_embedding, information, kg_entities_name, db_path, img_folder, entity_link_hash
                    )
                    information = reasoning_result["information"]

                    # Step 4: 如果需要，执行进一步推理
                    if reasoning_result["requires_further_reasoning"]:
                        further_reasoning_result = self.reasoning_engine.perform_further_reasoning(
                            question, question_embedding, reasoning_result, kg_entities_name, db_path, img_folder, entity_link_hash
                        )
                        information = further_reasoning_result["information"]

                    # Step 5: 组织信息
                    useful_information = self.info_organizer.organize_information(
                        question, information["entity_info"], information["table_info"], information["image_qa_info"]
                    )

                else:
                    # Step 6: 如果需要，对生成的答案进行进一步判断
                    if answer.lower() != 'yes' and answer.lower() != 'no' and answer.upper() in kg_entities_name:
                        answer = self.answer_judge.judge_answer(
                            question, answer.upper(), self.reasoning_engine.kg, kg_entities_name, entity_link_hash, img_folder
                        )
                        self.logger.log_info(f'Final Answer:\n{answer}')
                    return answer
            else:
                # Step 7: 重新进行推理
                reasoning_result = self.reasoning_engine.perform_reasoning(
                    question, question_embedding, information, kg_entities_name, db_path, img_folder, entity_link_hash
                )
                information = reasoning_result["information"]

                # 执行进一步推理
                if reasoning_result["requires_further_reasoning"]:
                    further_reasoning_result = self.reasoning_engine.perform_further_reasoning(
                        question, question_embedding, reasoning_result, kg_entities_name, db_path, img_folder, entity_link_hash
                    )
                    information = further_reasoning_result["information"]

                # 组织信息
                useful_information = self.info_organizer.organize_information(
                    question, information["entity_info"], information["table_info"], information["image_qa_info"]
                )
