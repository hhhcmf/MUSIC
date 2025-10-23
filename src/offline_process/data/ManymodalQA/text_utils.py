# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Text Utilities for LLM."""
import json
import sqlite3
from collections.abc import Iterator
from itertools import islice
import re
import tiktoken
import os
import argparse
import concurrent.futures
from utils.utils import read_jsonl_file
from config.config import config_loader
from utils.utils import write_json
from models.LLM.ModelFactory import ModelFactory
from prompt.prompt_generator import TextEntitiesPrompt
from utils.proprecss_llm_answer import extract_llm_text_entities_new
from utils.utils import combine_dictionaries, write_json, simply_combine_dict
from data.ManymodalQA.data_preprocess import get_dev_information

error_text_list, error_question_list = [], []
text_entities_result ={}

def num_tokens(text: str, token_encoder: tiktoken.Encoding | None = None) -> int:
    """Return the number of tokens in the given text."""
    if token_encoder is None:
        token_encoder = tiktoken.get_encoding("cl100k_base")
    return len(token_encoder.encode(text))  # type: ignore

def batched(iterable: Iterator, n: int):
    """
    Batch data into tuples of length n. The last batch may be shorter.

    Taken from Python's cookbook: https://docs.python.org/3/library/itertools.html#itertools.batched
    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        value_error = "n must be at least one"
        raise ValueError(value_error)
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

def chunk_text(
    text: str, max_tokens: int, token_encoder: tiktoken.Encoding | None = None
):
    """Chunk text by token length, ensuring that sentences are not split."""
    if token_encoder is None:
        token_encoder = tiktoken.get_encoding("cl100k_base")
    tokens = token_encoder.encode(text)  # type: ignore
    chunks = []
    current_chunk = []
    current_length = 0

    # Split text into sentences to avoid splitting mid-sentence
    sentences = re.split(r'(?<=[。！？.!?])', text)
    
    for sentence in sentences:
        sentence_tokens = token_encoder.encode(sentence)
        if current_length + len(sentence_tokens) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence_tokens
            current_length = len(sentence_tokens)
        else:
            current_chunk.extend(sentence_tokens)
            current_length += len(sentence_tokens)

    if current_chunk:
        chunks.append(current_chunk)

    yield from (token_encoder.decode(chunk) for chunk in chunks)


def llm_entities(text, text_id, args, LLM):
    print('进入提取实体')
    entities = {}
    max_num_tries = args.max_num_tries  
    count = 0
    flag = True
    print(f'------------{text_id}------------')
    while max_num_tries:
        print(max_num_tries)
        max_num_tries -= 1
        try:
            LLM.init_message()
            prompt_generator = TextEntitiesPrompt()
            prompt = prompt_generator.create_prompt(text)
            # print(prompt)
            result = LLM.predict(args, prompt)   
            # print(prompt)
            # print(f'---------result------\n{result}') 
            # print(result)
            text_entities = extract_llm_text_entities_new(result)
            # print('--------text entities-------------')
            # print(text_entities)
            entities = simply_combine_dict(text_entities, entities)
            # print('-------combine entities--------------')
            # print(entities)
        except Exception as e:
            # print(f"Error processing {text_id}: {e}")
            count +=1
            if count>3 and max_num_tries>0 and not entities:
                print(f"{text} Extract entities error occurred: {e}")
                max_num_tries = 0
                flag = False
                LLM.clear_cuda_memory()
                return entities, flag
            elif count<=3:
                max_num_tries += 1
            else:
                LLM.clear_cuda_memory()
                return entities, flag
    LLM.clear_cuda_memory()
    return entities, flag

def extract_text_entities(text, text_id, args, model):
    entities = {}
    entities, flag = llm_entities(text, text_id, args, model)
    print(entities)
    if not entities:
        flag = False
    return entities, flag

#TODO title entities
# def get_title_entities(data, args, model):


def get_texts_entities_result(data, args):#text_data:id、title、text  
    model = ModelFactory.create_model(args)
    result = []
    data_id = data['id']
    
    if data.get('image'):
        caption = data['image']['caption']
        caption_entities, image_caption_flag = extract_text_entities(caption, data_id, args, model)
        if image_caption_flag:
            
            text_entities_result[data_id] ={'id':data_id, 'type':'image', 'text': caption, 'entities':caption_entities}

            result.append({'id':data_id, 'type':'image', 'text': caption, 'entities':caption_entities})

    
    if data.get('text'):
        text = data['text']
        chunk_entities, chunk_flag = extract_text_entities(text, data_id, args, model)
        if chunk_flag:
            text_entities_result[data_id] ={'id':data_id, 'type':'text', 'text': text, 'entities':chunk_entities}
            result.append({'id':data_id, 'type':'text', 'text': text, 'entities':chunk_entities})
    return result

# def process_texts_concurrently(texts, args):
#     with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
#         future_to_text = {executor.submit(get_texts_entities_result, text, args): text for text in texts}
#         for future in concurrent.futures.as_completed(future_to_text):
#             text_data = future_to_text[future]
#             try:
#                 result = future.result()
#                 print(f"Text with ID {text_data['id']} generated")
#             except Exception as exc:
#                 error_text_list.append(text_data)
#                 print(f"Text with ID {text_data['id']} generated an exception: {exc}")

def write_entities_to_db(conn, entities_data_list):
    for entities_data in entities_data_list:
        cursor = conn.cursor()
      
        batch_insert = []
        # print('Initial batch_insert:', batch_insert)
        try:
            # Head entities
            for key, value in entities_data["entities"].items():
                if entities_data['type'] == 'text':
                    isTextTitle = 1
                    isImageTitle = 0
                else:
                    isImageTitle = 1
                    isTextTitle = 0
                entity_name = key
                entity_type = value.get('Type')
                description = value.get("Description")
                batch_insert.append(
                    (entities_data["id"], entity_name, entity_type, entities_data['type'], '', description,
                        entities_data['text'], isTextTitle, isImageTitle))

           
            # print('Final batch_insert:', batch_insert)

            # Executing batch insert
            cursor.executemany(
                'INSERT OR IGNORE INTO entities (data_id, entity, entity_type, type, title, description, text, '
                'isTextTitle, isImageTitle) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', batch_insert)

            # print('Batch insert executed successfully.')

            # Committing transaction
            conn.commit()
            # print('11111: Commit successful')
            return True, ''
        
        except Exception as e:
            print(f"Error occurred during database insert or commit: {e}")
            return False, e
        
        finally:
            cursor.close()



def process_texts_concurrently(texts, args):
    results = []

    # Connect to SQLite database
    # print(f'{args.dataset}_kg_{args.engine}.db')
    conn = sqlite3.connect(f'{args.dataset}_kg_{args.engine}.db')
    
    # Create the table if it does not exist
    conn.execute('''
    CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data_id TEXT,
        entity TEXT,
        entity_type TEXT,
        type TEXT,
        title TEXT,
        description TEXT,
        text TEXT, 
        isTextTitle INTEGER DEFAULT 0, 
        isImageTitle INTEGER DEFAULT 0
    )
    ''')
    conn.commit()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_text = {executor.submit(get_texts_entities_result, text, args): text for text in texts}

        for future in concurrent.futures.as_completed(future_to_text):
            text_data = future_to_text[future]#是输入
            # try:
            result = future.result()
            results.append(result)
            print(result)
            print(f"Text with ID {text_data['id']} processed successfully.")
            # flag, message = write_entities_to_db(conn, result)
            # if flag:
            #     print(f"Text with ID {text_data['id']} processed successfully.")
            # else:
            #     error_text_list.append(text_data)
            #     print(f"Text with ID {text_data['id']} generated an exception: {message}")
            # except Exception as exc:
            #     error_text_list.append(text_data)
            #     print(f"Text with ID {text_data['id']} generated an exception: {exc}")

    # Close the database connection
    conn.close()
    return results, error_text_list

def extract_WEBQA_text_entites(args):
    question_datas_list =  get_dev_information()
    data_list = []
    data_id_list = []
    data = {}
    for question_data in question_datas_list:
        data_id = question_data['id']

        if question_data['image']:
            data_list.append({'id':data_id, 'image':question_data['image']})
            data_id_list.append(data_id)
            data[data_id] = {'id':data_id, 'image':question_data['image']}

        text = question_data['text']
        chunks = list(chunk_text(text, args.max_tokens_per_chunk))
        for idx, chunk in enumerate(chunks):
            chunk_id = f'{data_id}_{idx}'
            data_list.append({'id':chunk_id, 'text':chunk})
            data_id_list.append(chunk_id)
            data[chunk_id] ={}
            data[chunk_id] = {'id':chunk_id, 'text':chunk}
            
    all_data = list(set(data_id_list))
    conn = sqlite3.connect(f'{args.dataset}_kg_{args.engine}.db')
    cursor = conn.cursor()
    cursor.execute('select distinct(data_id) from entities')
    used_data_list = [row[0] for row in cursor.fetchall()]
    data_id_list = list(set(data_id_list) - set(used_data_list))
    cursor.close()
    conn.close()
    for d_id in data_id_list:
        text_data = data[d_id]
        data_list.append(text_data)
    print(f'未处理:{len(data_list)},已处理{len(all_data)/len(data_list)}')
    results, error_text_list = process_texts_concurrently(data_list,args)
   

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--temperature", type=float, default=0.2, help=""
    )
    parser.add_argument(
        '--dataset', default='manymodalqa',
        help="dataset",
        choices=["mmqa", "webqa","manymodalqa" ]
    )
    parser.add_argument(
        "--max_tokens_per_chunk", type=int, default=600, help=""
    )
    parser.add_argument(
        "--engine", default='llama3-8b', help="llama2-7b, llama3-8b, gpt-3.5-turbo, qwen-turbo, qwen-plus",
        choices=["llama2-7b", "llama3-8b", "gpt-3.5-turbo", 'qwen-turbo', 'qwen-plus', 'qwen-max', 'qwen-plus-latest']
    )
    parser.add_argument(
        "--model_path", default='/home/cmf/era-cot-main/Llama-2-7b-chat-hf', help="your local model path"
    )
    parser.add_argument(
        "--api_key", default='sk-llqpW9khKdoFanaN1e992c3077Bd4a05A9D581BcA382F889', help="your api_key"
    )
    parser.add_argument(
        "--api_base", default='https://api.xiaoai.plus/v1', help="api_base"
    )
    parser.add_argument(
        "--start", type=int, default=0, help='number'
    )
    parser.add_argument(
        "--end",  type=int, default=3055, help='number, max7540'
    )
    parser.add_argument(
        '--max_num_tries',  type=int, default=5, help="llm max tries"
    )
    parser.add_argument(
        '--max_workers',  type=int, default=5, help="max workers num"
    )
    parser.add_argument(
        '--save_folder', default='/home/cmf/multiQA/data/ManymodalQA/data/kg', help="save_folder"
    )
    parsed_args = parser.parse_args()
    return parsed_args

def main(args):
    if args.dataset == 'manymodalqa':
        extract_WEBQA_text_entites(args)
    entity_outpath = os.path.join(args.save_folder, f'{args.dataset}_text_entities_{args.engine}.json')
    write_json(entity_outpath, text_entities_result)
    

if __name__ == '__main__':
    print('开始')
    args = parse_arguments()
    
    print(args.engine)
    main(args)

# if __name__ == "__main__":
#     args = parse_arguments()
#     max_tokens_per_chunk = 600
#     # 1200:19 600:24

#     # 对文本进行分块
#     chunks = list(chunk_text(long_text, max_tokens_per_chunk))
#     LLM = ModelFactory.create_model(args)
#     # 打印每个分块
#     print(chunks[0])
#     entities, flag = llm_entities(chunks[0], LLM, args, 0)
#     print(entities)
#     print(len(entities))
#     # 示例文本
#     long_text = '''
#     "\nA broch ( /\u02c8br\u0252x/) is an Iron Age drystone hollow-walled structure found in Scotland. Brochs belong to the classification \"complex atlantic roundhouse\" devised by Scottish archaeologists in the 1980s. Their origin is a matter of some controversy.\nThe word broch is derived from Lowland Scots 'brough', meaning (among other things) fort. In the mid-19th century Scottish antiquaries called brochs 'burgs', after Old Norse borg, with the same meaning. Place names in Scandinavian Scotland such as Burgawater and Burgan show that Old Norse borg is the older word used for these structures in the north. Brochs are often referred to as duns in the west. Antiquarians began to use the spelling broch in the 1870s.\nA precise definition for the word has proved elusive. Brochs are the most spectacular of a complex class of roundhouse buildings found throughout Atlantic Scotland. The Shetland Amenity Trust lists about 120 sites in Shetland as candidate brochs, while the Royal Commission on the Ancient and Historical Monuments of Scotland (RCAHMS) identifies a total of 571 candidate broch sites throughout the country. Researcher Euan MacKie has proposed a much smaller total for Scotland of 104.\nThe origin of brochs is a subject of continuing research. Sixty years ago most archaeologists believed that brochs, usually regarded as the 'castles' of Iron Age chieftains, were built by immigrants who had been pushed northward after being displaced first by the intrusions of Belgic tribes into what is now southeast England at the end of the second century BC and later by the Roman invasion of southern Britain beginning in AD 43. Yet there is now little doubt that the hollow-walled broch tower was purely an invention in what is now Scotland; even the kinds of pottery found inside them that most resembled south British styles were local hybrid forms. The first of the modern review articles on the subject (MacKie 1965) did not, as is commonly believed, propose that brochs were built by immigrants, but rather that a hybrid culture formed from the blending of a small number of immigrants with the native population of the Hebrides produced them in the first century BC, basing them on earlier, simpler, promontory forts. This view contrasted, for example, with that of Sir W. Lindsay Scott, who argued, following Childe (1935), for a wholesale migration into Atlantic Scotland of people from southwest England.\nMacKie's theory has fallen from favour too, mainly because starting in the 1970s there was a general move in archaeology away from 'diffusionist' explanations towards those pointing to exclusively indigenous development. Meanwhile, the increasing number \u2013 albeit still pitifully few \u2013 of radiocarbon dates for the primary use of brochs (as opposed to their later, secondary use) still suggests that most of the towers were built in the 1st centuries BC and AD. A few may be earlier, notably the one proposed for Old Scatness Broch in Shetland, where a sheep bone dating to 390\u2013200 BC has been reported.\nThe other broch claimed to be substantially older than the 1st century BC is Crosskirk in Caithness, but a recent review of the evidence suggests that it cannot plausibly be assigned a date earlier than the 1st centuries BC/AD\nThe distribution of brochs is centred on northern Scotland. Caithness, Sutherland and the Northern Isles have the densest concentrations, but there are a great many examples in the west of Scotland and the Hebrides. Although mainly concentrated in the northern Highlands and the Islands, a few examples occur in the Borders (for example Edin's Hall Broch and Bow Castle Broch); on the west coast of Dumfries and Galloway; and near Stirling. In a c.1560 sketch there appears to be a broch by the river next to Annan Castle in Dumfries and Galloway. This small group of southern brochs has never been satisfactorily explained.\nThe original interpretation of brochs, favoured by nineteenth century antiquarians, was that they were defensive structures, places of refuge for the community and their livestock. They were sometimes regarded as the work of Danes or Picts. From the 1930s to the 1960s, archaeologists such as V. Gordon Childe and later John Hamilton regarded them as castles where local landowners held sway over a subject population.\nThe castle theory fell from favour among Scottish archaeologists in the 1980s, due to a lack of supporting archaeological evidence. These archaeologists suggested defensibility was never a major concern in the siting of a broch, and argued that they may have been the \"stately homes\" of their time, objects of prestige and very visible demonstrations of superiority for important families (Armit 2003). Once again, however, there is a lack of archaeological proof for this reconstruction, and the sheer number of brochs, sometimes in places with a lack of good land, makes it problematic.\nBrochs' close groupings and profusion in many areas may indeed suggest that they had a primarily defensive or even offensive function. Some of them were sited beside precipitous cliffs and were protected by large ramparts, artificial or natural: a good example is at Burland near Gulberwick in Shetland, on a clifftop and cut off from the mainland by huge ditches. Often they are at key strategic points. In Shetland they sometimes cluster on each side of narrow stretches of water: the broch of Mousa, for instance, is directly opposite another at Burraland in Sandwick. In Orkney there are more than a dozen on the facing shores of Eynhallow Sound, and many at the exits and entrances of the great harbour of Scapa Flow. In Sutherland quite a few are placed along the sides and at the mouths of deep valleys. Writing in 1956 John Stewart suggested that brochs were forts put up by a military society to scan and protect the countryside and seas.\nFinally, some archaeologists consider broch sites individually, doubting that there ever was a single common purpose for which every broch was constructed. There are differences between the various areas in which brochs are found, with regard to position, dimensions and likely status. For example, the broch \"villages\" which occur at a few places in Orkney have no parallel in the Western Isles.\nGenerally, brochs have a single entrance with bar-holes, door-checks and lintels. There are mural cells and there is a scarcement (ledge), perhaps for timber-framed lean-to dwellings lining the inner face of the wall. Also there is a spiral staircase winding upwards between the inner and outer wall and connecting the galleries. Brochs vary from 5 to 15\u00a0metres (16\u201350\u00a0ft) in internal diameter, with 3\u00a0metre (10\u00a0ft) thick walls. On average, the walls only survive to a few metres in height. There are five extant examples of towers with significantly higher walls: Dun Carloway on Lewis, Dun Telve and Dun Troddan in Glenelg, Mousa in Shetland and Dun Dornaigil in Sutherland, all of whose walls exceed 6.5\u00a0m (21\u00a0ft) in height.\nMousa's walls are the best preserved and are still 13\u00a0m tall; it is not clear how many brochs originally stood this high. A frequent characteristic is that the walls are galleried: with an open space between, the outer and inner wall skins are separate but tied together with linking stone slabs; these linking slabs may in some cases have served as steps to higher floors. It is normal for there to be a cell breaking off from the passage beside the door; this is known as the guard cell. It has been found in some Shetland brochs that guard cells in entrance passageways are close to large door-check stones. Although there was much argument in the past, it is now generally accepted among archaeologists that brochs were roofed, perhaps with a conical timber framed roof covered with a locally sourced thatch. The evidence for this assertion is still fairly scanty, although excavations at Dun Bharabhat, Lewis, may support it. The main difficulty with this interpretation continues to be the potential source of structural timber, though bog and driftwood may have been plentiful sources.\nOn the islands of Orkney and Shetland there are very few cells at ground floor. Most brochs have scarcements (ledges) which may have allowed the construction of a very sturdy wooden first floor (first spotted by the antiquary George Low in Shetland in 1774), and excavations at Loch na Berie on the Isle of Lewis show signs of a further, second floor (e.g. stairs on the first floor, which head upwards). Some brochs such as Dun Dornaigil and Culswick in Shetland have unusual triangular lintels above the entrance door.\nAs in the case of Old Scatness in Shetland (near Jarlshof and Burroughston on Shapinsay), brochs were sometimes located close to arable land and a source of water (some have wells or natural springs rising within their central space). Sometimes, on the other hand, they were sited in wilderness areas (e.g. Levenwick and Culswick in Shetland, Castle Cole in Sutherland). Brochs are often built beside the sea (Carn Liath, Sutherland); sometimes they are on islands in lochs (e.g. Clickimin in Shetland).\nAbout 20 Orcadian broch sites include small settlements of stone buildings surrounding the main tower. Examples include Howe, near Stromness, Gurness Broch in the north west of Mainland, Orkney, Midhowe on Rousay and Lingro near Kirkwall (destroyed in the 1980s). There are \"broch village\" sites in Caithness, but elsewhere they are unknown.\nMost brochs are unexcavated. Those that have been properly examined show that they continued to be in use for many centuries, with the interiors often modified and changed, and that they underwent many phases of habitation and abandonment. The end of the broch period seems to have come around AD 200\u2013300.\nMousa, Old Scatness and Jarlshof: The Crucible of Iron Age Shetland is a combination of three broch sites in Shetland that are on the United Kingdom \"Tentative List\" of possible nominations for the UNESCO World Heritage Programme list of sites of outstanding cultural or natural importance to  the common heritage of humankind. \nThis list, published in July 2010, includes sites that may be nominated for inscription over the next 5\u201310 years.\n"
#         '''

    # 设置每个chunk的最大token数
    
    # for idx, chunk in enumerate(chunks):
    #     print(f"Chunk {idx + 1}:")
    #     print(chunk)
    #     print("-" * 20)
        
    