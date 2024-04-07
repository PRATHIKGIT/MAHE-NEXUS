from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
# from ctransformers import AutoModelForCausalLM

import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    model = Ollama(model="health")
    # def load_llm():
    #   llm = AutoModelForCausalLM.from_pretrained("chatdoctor.gguf",model_type='llama',max_new_tokens = 1096,repetition_penalty = 1.13,temperature = 0.1)
    #   return llm
    # model=load_llm()
    prompt = ChatPromptTemplate.from_messages(
             [
            (
                "system",
                "You're a very knowledgeable doctor.Provide just the answer related to medical field.",
            ),
            ("user", "{question}"),
                    ]
       )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()


