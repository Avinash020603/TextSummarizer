import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader



st.set_page_config(page_title="Summarize Text From YT or Website")
st.title("Summarize Text From YT or Website")
st.subheader('Summarize URL')




with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")
    language=st.text_input("Language")
    Word_limit=st.text_input("Word Limit")

generic_url=st.text_input("URL",label_visibility="collapsed")


llm = None
if groq_api_key.strip():
    try:
        llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
    except Exception as e:
        st.error("Invalid Groq API Key or model initialization failed.")

prompt_template="""
Provide a summary of the following content in {word_limit} words:
The summary should be in {language}
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text","language","word_limit"])

if st.button("Summarize the Content from YT or Website"):

    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")

    else:
        try:
            with st.spinner("Waiting..."):
              
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=False)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()

               
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary = chain.run({
                "input_documents": docs,
                "language": language,
                "word_limit": Word_limit
                })

                 st.success("Summary generated successfully!")

           
                st.write(output_summary)


                st.download_button(
                label="📥 Download Summary as .txt",
                data=output_summary,
                file_name="summary.txt",
                mime="text/plain"
            )


                st.success(output_summary)
        except Exception as e:
            st.exception(e)
                    
