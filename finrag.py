import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import urllib.request
import gradio as gr

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_pdf():
    # urllib.request.urlretrieve("https://github.com/chatgpt-kr/openai-api-tutorial/raw/main/ch07/2020_%EA%B2%BD%EC%A0%9C%EA%B8%88%EC%9C%B5%EC%9A%A9%EC%96%B4%20700%EC%84%A0_%EA%B2%8C%EC%8B%9C.pdf", filename="2020_경제금융용어 700선_게시.pdf")

    loader = PyPDFLoader("2020_경제금융용어 700선_게시.pdf")
    texts = loader.load_and_split()
    texts = texts[13:-1]

    return texts

def create_vectordb():
    # 기존 DB가 있는지 확인
    if os.path.exists("./chroma_db"):
        print("기존 벡터 DB를 로드합니다...")
        embedding = OpenAIEmbeddings(chunk_size=100)
        vectordb = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embedding
        )
        print(f"로드된 문서 수: {vectordb._collection.count()}")
        return vectordb
    
    # 기존 DB가 없으면 새로 생성
    print("새로운 벡터 DB를 생성합니다...")
    texts = load_pdf()
    embedding = OpenAIEmbeddings(chunk_size=100)

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory="./chroma_db"  # DB를 현재 폴더에 저장
    )
    
    # DB를 디스크에 저장 (Chroma 0.4.x에서는 자동으로 저장됨)
    # vectordb.persist()  # 이 줄 제거

    print(f"생성된 문서 수: {vectordb._collection.count()}")
    return vectordb

def create_retriever():
    vectordb = create_vectordb()
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    return retriever

def create_chain():

    template = """당신은 ABC은행에서 만든 금융 용어를 설명해주는 금융쟁이입니다.
    주어진 검색 결과를 바탕으로 답변하세요.
    검색 결과에 없는 내용이라면 답변할 수 없다고 하세요. 반말로 친근하게 답변하세요.
    
    Context: {context}

    Question: {question}
    Answer:
    """

    retriever = create_retriever()  # 기존 retriever 재사용
    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # retriever 결과를 문자열로 변환하는 함수
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 전체 chain을 처리하는 함수
    def create_rag_chain(input_dict):
        question = input_dict["question"]
        docs = retriever.invoke(question)  # get_relevant_documents 대신 invoke 사용
        context = format_docs(docs)
        return {"context": context, "question": question}

    chain = (
        RunnableLambda(create_rag_chain)
        | prompt 
        | llm 
        | StrOutputParser()
    )

    return chain

def create_ui(chain):
    # 인터페이스 생성
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(label="경제금융용어 챗봇", type="messages") # 챗봇 레이블을 좌측 상단에 구성
        msg = gr.Textbox(label="질문해주세요!")  # 하단의 채팅창 레이블
        clear = gr.Button("대화 초기화")  # 대화 초기화 버튼

        # 챗봇의 답변을 처리하는 함수
        def respond(message, chat_history):
            result = chain.invoke({"question": message})
            bot_message = result  # chain이 직접 문자열을 반환하므로 result['result'] 불필요

            # 채팅 기록에 사용자의 메시지와 봇의 응답을 추가 (새로운 messages 형식)
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": bot_message})
            return "", chat_history

        # 사용자의 입력을 제출(submit)하면 respond 함수가 호출
        msg.submit(respond, [msg, chatbot], [msg, chatbot])

        # '초기화' 버튼을 클릭하면 채팅 기록을 초기화
        clear.click(lambda: None, None, chatbot, queue=False)

    # 인터페이스 실행
    demo.launch(debug=True)

def main():
    chain = create_chain()

    questions = ["디커플링이란 무엇인가?", "너는 뭘하는 챗봇이니?"]

    results = chain.batch([{"question":q} for q in questions])
    for q, r in zip(questions, results):
        print(f"\n질문: {q}")
        print(f"답변: {r}")

    # for question in questions:
    #     print(f"\n질문: {question}")
    #     try:
    #         result = chain.invoke({"question": question})
    #         print(f"답변: {result}")
    #     except Exception as e:
    #         print(f"에러 발생: {e}")

if __name__ == "__main__":
    # main()
    create_ui(create_chain())
