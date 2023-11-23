from langchain.document_loaders import TextLoader
from langchain.agents import Tool, AgentType
from dotenv import load_dotenv
from os import getenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.utilities import GoogleSerperAPIWrapper
from operator import itemgetter
from langchain.chat_models import ChatOpenAI 
from langchain.agents import initialize_agent
from langchain.schema.messages import SystemMessage
# from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

PREFIX = """Kamu adalah ChatBOTPolitic yang bertugas untuk menjawab pertanyaan seputar partai politik di Indonesia.
Kamu HARUS menjawab dengan BAHASA INDONESIA apapun kondisinya. Tidak boleh sama sekali menggunakan bahasa selain BAHASA INDONESIA.
Kamu HARUS menjawab pertanyaan HANYA berdasarkan data yang diberikan saja. Tidak boleh menjawab pertanyaan dari luar sumber data yang diberikan.
Jika data yang diberikan tidak ada, maka kamu harus mencari jawabannya di semua 'tools' yang ada. 
Jika setelah mencari di semua tools tetap tidak ada, anda harus menjawab 'Maaf, saya tidak dapat menemukan informasi yang dicari karena keterbatasan informasi.'.
Gunakan Bahasa Indonesia dalam menjawab pertanyaan dari penanya. Jika ada pertanyaan yang diluar topik Partai Politik, kamu harus menolak untuk menjawab pertanyaan tersebut."""

def split_texts(text_name : str,):
  loader = TextLoader(text_name, encoding="utf-8")
  documents = loader.load()
  texts = documents
  return texts[0].page_content

llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.0, openai_api_key=getenv("OPENAI_API_KEY"))
# embeddings = OpenAIEmbeddings()

def pan_docsearch(query):
    return split_texts("./data/data-txt/Partai Amanat Nasional (PAN).txt")

def pbb_docsearch(query):
    return split_texts("./data/data-txt/Partai Bulan Bintang (PBB).txt")
    
def partai_buruh_docsearch(query):
    return split_texts("./data/data-txt/Partai Buruh.txt")
    
def pdip_docsearch(query):
    return split_texts("./data/data-txt/Partai Demokrasi Indonesia Perjuangan (PDI-P).txt")    

def partai_demokrat_docsearch(query):
    return split_texts("./data/data-txt/Partai Demokrat.txt")

def pgaruda_docsearch(query):
    return split_texts("./data/data-txt/Partai Garda Perubahan Indonesia (Garuda).txt")

def pgelora_docsearch(query):
    return split_texts("./data/data-txt/Partai Gelombang Rakyat Indonesia (Gelora).txt")

def pgerindra_docsearch(query):
    return split_texts("./data/data-txt/Partai Gerakan Indonesia Raya (Gerindra).txt")
    
def pgolkar_docsearch(query):
    return split_texts("./data/data-txt/Partai Golongan Karya (Golkar).txt")

def phanura_docsearch(query):
    return split_texts("./data/data-txt/Partai Hati Nurani Rakyat (Hanura).txt")
    
def pks_docsearch(query):
    return split_texts("./data/data-txt/Partai Keadilan Sejahtera (PKS).txt")
    
def pkb_docsearch(query):
    return split_texts("./data/data-txt/Partai Kebangkitan Bangsa (PKB).txt")
    
def pkn_docsearch(query):
    return split_texts("./data/data-txt/Partai Kebangkitan Nusantara (PKN).txt")

def pnasdem_docsearch(query):
    return split_texts("./data/data-txt/Partai Nasional Demokrat (NasDem).txt")
    
def pperindo_docsearch(query):
    return split_texts("./data/data-txt/Partai Perindo.txt")

def ppp_docsearch(query):
    return split_texts("./data/data-txt/Partai Persatuan Pembangunan (PPP).txt")

def psi_docsearch(query):
    return split_texts("./data/data-txt/Partai Solidaritas Indonesia (PSI).txt")

def pu_docsearch(query):
    return split_texts("./data/data-txt/Partai Ummat (PU).txt")

def party_query_tool(input_text):
    concatenated_text = f"{input_text} \n Partai apa yang diikuti oleh orang tersebut?"

    search = GoogleSerperAPIWrapper(gl="id", hl="id")
    results = search.results(concatenated_text)
    get_title = map(itemgetter('title'), results['organic'][:3])
    get_snippet = map(itemgetter('snippet'), results['organic'][:3])

    concatenate_zip = zip(get_title, get_snippet)
    concatenate_map = map(lambda x: f"{x[0]} {x[1]}", concatenate_zip)
    concatenated = ' '.join(concatenate_map)

    query_template = PromptTemplate.from_template("Siapakah dia dan partai apa yang diikuti oleh orang tersebut jika data yang diberikan ada pada 'Context'?"
                                                  "Context: '{concatenated}'")

    llm_party = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
    llm_chain = LLMChain(llm=llm_party, prompt=query_template)
    query = llm_chain(concatenated)["text"]
    query.split(" ")

    agent_kwargs = {
            "system_message" : SystemMessage(content=PREFIX),
        }

    prompt_results = initialize_agent(
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=tools[:-1],
        llm=llm,
        verbose=True,
        max_iterations=5,
        agent_kwargs=agent_kwargs,
        # memory=memory

    )
    
    results = prompt_results(query)["output"]
    return results

tools = [    
    Tool(
        name="Partai_Amanat_Nasional_PAN",
        func=pan_docsearch,
        # func=split_texts("../data/data-txt/Partai Amanat Nasional (PAN).txt"),
        # func=meaning_of_life,
        description=("Berguna untuk ketika kamu memerlukan informasi seputar Partai Amanat Nasional (PAN). Masukan harus berupa pertanyaan yang berhubungan dengan Partai Amanat Nasional PAN"
                     "Terdapat informasi seperti: profil partai, inti dari partai, ideologi partai, tokoh-tokoh partai, fakta unik partai, rekam jejak partai, korupsi yang dilakukan kader partai, mantan narapidana, dan sebagainya")

    ),
    Tool(
        name="Partai_Bulan_Bintang_PBB",
        func=pbb_docsearch,
        # func=split_texts("../data/data-txt/Partai Bulan Bintang (PBB).txt"),
        description=("Berguna untuk ketika kamu memerlukan informasi seputar Partai Bulan Bintang (PBB). Masukan harus berupa pertanyaan yang berhubungan dengan Partai Bulan Bintang PBB"
                     "Terdapat informasi seperti: profil partai, inti dari partai, ideologi partai, tokoh-tokoh partai, fakta unik partai, rekam jejak partai, korupsi yang dilakukan kader partai, mantan narapidana, dan sebagainya")
    ),
    Tool(
        name="Partai_Buruh",
        func=partai_buruh_docsearch,
        # func=split_texts("../data/data-txt/Partai Buruh.txt"),
        description=("Berguna untuk ketika kamu memerlukan informasi seputar Partai Buruh. Masukan harus berupa pertanyaan yang berhubungan dengan Partai Buruh"
                     "Terdapat informasi seperti: profil partai, inti dari partai, ideologi partai, tokoh-tokoh partai, fakta unik partai, rekam jejak partai, korupsi yang dilakukan kader partai, mantan narapidana, dan sebagainya")
    ),
    Tool(
        name="Partai_Demokrasi_Indonesia_Perjuangan_PDI-P",
        func=pdip_docsearch,
        # func=split_texts("../data/data-txt/Partai Demokrasi Indonesia Perjuangan (PDI-P).txt"),
        description=("Berguna untuk ketika kamu memerlukan informasi seputar Partai Demokrasi Indonesia Perjuangan (PDIP). Masukan harus berupa pertanyaan yang berhubungan dengan Partai Demokrasi Indonesia Perjuangan PDIP"
                     "Terdapat informasi seperti: profil partai, inti dari partai, ideologi partai, tokoh-tokoh partai, fakta unik partai, rekam jejak partai, korupsi yang dilakukan kader partai, mantan narapidana, dan sebagainya")
    ),
    Tool(
        name="Partai_Demokrat",
        func=partai_demokrat_docsearch,
        description=("Berguna untuk ketika kamu memerlukan informasi seputar Partai Demokrat. Masukan harus berupa pertanyaan yang berhubungan dengan Partai Demokrat"
    "Terdapat informasi seperti: profil partai, inti dari partai, ideologi partai, tokoh-tokoh partai, fakta unik partai, rekam jejak partai, korupsi yang dilakukan kader partai, mantan narapidana, dan sebagainya")
    ),
    Tool(
        name="Partai_Garda_Perubahan_Indonesia_Garuda",
        func=pgaruda_docsearch,
        description=("Berguna untuk ketika kamu memerlukan informasi seputar Partai Garda Perubahan Indonesia (Partai Garuda). Masukan harus berupa pertanyaan yang berhubungan dengan Partai Garuda"
    "Terdapat informasi seperti: profil partai, inti dari partai, ideologi partai, tokoh-tokoh partai, fakta unik partai, rekam jejak partai, korupsi yang dilakukan kader partai, mantan narapidana, dan sebagainya")
    ),
    Tool(
        name="Partai_Gelombang_Rakyat_Indonesia_Gelora",
        func=pgelora_docsearch,
        description=("Berguna untuk ketika kamu memerlukan informasi seputar Partai Gelombang Rakyat Indonesia (Partai Gelora). Masukan harus berupa pertanyaan yang berhubungan dengan Partai Gelora"
    "Terdapat informasi seperti: profil partai, inti dari partai, ideologi partai, tokoh-tokoh partai, fakta unik partai, rekam jejak partai, korupsi yang dilakukan kader partai, mantan narapidana, dan sebagainya")
    ),
    Tool(
        name="Partai_Gerakan_Indonesia_Raya_Gerindra",
        func=pgerindra_docsearch,
        description=("Berguna untuk ketika kamu memerlukan informasi seputar Partai Gerakan Indonesia Raya (Partai Gerindra). Masukan harus berupa pertanyaan yang berhubungan dengan Partai Gerakan Indonesia Raya Gerindra"
    "Terdapat informasi seperti: profil partai, inti dari partai, ideologi partai, tokoh-tokoh partai, fakta unik partai, rekam jejak partai, korupsi yang dilakukan kader partai, mantan narapidana, dan sebagainya")
    ),
    Tool(
        name="Partai_Golongan_Karya_Golkar",
        func=pgolkar_docsearch,
        description=("Berguna untuk ketika kamu memerlukan informasi seputar Partai Golongan Karya (Partai Golkar). Masukan harus berupa pertanyaan yang berhubungan dengan Partai Golongan Karya Golkar"
    "Terdapat informasi seperti: profil partai, inti dari partai, ideologi partai, tokoh-tokoh partai, fakta unik partai, rekam jejak partai, korupsi yang dilakukan kader partai, mantan narapidana, dan sebagainya")
    ),
    Tool(
        name="Partai_Hati_Nurani_Rakyat_Hanura",
        func=phanura_docsearch,
        description=("Berguna untuk ketika kamu memerlukan informasi seputar Partai Hati Nurani Rakyat (Partai Hanura). Masukan harus berupa pertanyaan yang berhubungan dengan Partai Hati Nurani Rakyat Hanura"
    "Terdapat informasi seperti: profil partai, inti dari partai, ideologi partai, tokoh-tokoh partai, fakta unik partai, rekam jejak partai, korupsi yang dilakukan kader partai, mantan narapidana, dan sebagainya")
    ),
    Tool(
        name="Partai_Keadilan_Sejahtera_PKS",
        func=pks_docsearch,
        description=("Berguna untuk ketika kamu memerlukan informasi seputar Partai Keadilan Sejahtera (PKS). Masukan harus berupa pertanyaan yang berhubungan dengan Partai Keadilan Sejahtera PKS"
    "Terdapat informasi seperti: profil partai, inti dari partai, ideologi partai, tokoh-tokoh partai, fakta unik partai, rekam jejak partai, korupsi yang dilakukan kader partai, mantan narapidana, dan sebagainya")
    ),
    Tool(
        name="Partai_Kebangkitan_Bangsa_PKB",
        func=pkb_docsearch,
        description=("Berguna untuk ketika kamu memerlukan informasi seputar Partai Kebangkitan Bangsa (PKB). Masukan harus berupa pertanyaan yang berhubungan dengan Partai Kebangkitan Bangsa PKB"
    "Terdapat informasi seperti: profil partai, inti dari partai, ideologi partai, tokoh-tokoh partai, fakta unik partai, rekam jejak partai, korupsi yang dilakukan kader partai, mantan narapidana, dan sebagainya")
    ),
    Tool(
        name="Partai_Kebangkitan_Nusantara_PKN",
        func=pkn_docsearch,
        description=("Berguna untuk ketika kamu memerlukan informasi seputar Partai Kebangkitan Nusantara (PKN). Masukan harus berupa pertanyaan yang berhubungan dengan Partai Kebangkitan Nusantara PKN"
    "Terdapat informasi seperti: profil partai, inti dari partai, ideologi partai, tokoh-tokoh partai, fakta unik partai, rekam jejak partai, korupsi yang dilakukan kader partai, mantan narapidana, dan sebagainya")
    ),
    Tool(
        name="Partai_Nasional_Demokrat_NasDem",
        func=pnasdem_docsearch,
        # func=split_texts("../data/data-txt/Partai Nasional Demokrat (NasDem).txt"),
        description=("Berguna untuk ketika kamu memerlukan informasi seputar Partai Nasional Demokrat (Partai NasDem). Masukan harus berupa pertanyaan yang berhubungan dengan Partai Nasional Demokrat NasDem"
                     "Terdapat informasi seperti: profil partai, inti dari partai, ideologi partai, tokoh-tokoh partai, fakta unik partai, rekam jejak partai, korupsi yang dilakukan kader partai, mantan narapidana, dan sebagainya")
    ),

    Tool(
        name="Orang_Dalam_Partai",
        func=party_query_tool,
        description=(
            "Ketika ada pertanyaan yang tidak diikuti dengan keterangan partai-nya dari mana, maka kamu dapat menggunakan tool ini."
        )
    )
]



class PoliticBotAgent(object):
    def __init__(self):
        self.memory = None
        self.agent_chain = None
#         self.id = str(id)

        self.initialize_agent()

    def initialize_agent(self):
        agent_kwargs = {
            "system_message" : SystemMessage(content=PREFIX),
        }

        self.agent_chain = initialize_agent(
            agent=AgentType.OPENAI_FUNCTIONS,
            tools=tools,
            llm=llm,
            verbose=True,
            max_iterations=18,
            agent_kwargs=agent_kwargs,
            # memory=memory
        )
    
    def async_generate(self, text):
        response = self.agent_chain(text)
        return response["output"]