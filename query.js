import * as dotenv from 'dotenv';
dotenv.config();

import { GoogleGenAI } from "@google/genai";


import readlineSync from 'readline-sync';

const ai = new GoogleGenAI({});
const History = []
async function transformQuery(question){

History.push({
    role:'user',
    parts:[{text:question}]
    })  

const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
    Only output the rewritten question and nothing else.
      `,
    },
 });
 
 History.pop()
 
 return response.text


}

async function chatting(userProblem) {
    // const { ChatOpenAI } = await import('@langchain/openai');
    // const { PineconeStore } = await import('@langchain/pinecone');
    
    // const model = new ChatOpenAI({
    //     temperature: 0,
    //     modelName: 'gpt-3.5-turbo',
    // });
    
    // const vectorStore = await PineconeStore.fromExistingIndex({
    //     pineconeIndexName: process.env.PINECONE_INDEX_NAME,
    //     pineconeNamespace: 'dsa-pdf',
    //     embeddings: new GoogleGenerativeAIEmbeddings({
    //     apiKey: process.env.GEMINI_API_KEY,
    //     model: 'text-embedding-004',
    //     }),
    // });
    
    // const chain = vectorStore.asRetrievalQAChain(model, {
    //     returnSourceDocuments: true,
    // });
    
    // const response = await chain.invoke({ query: userProblem });
    // console.log(response);
     
    // If the user question is a follow-up question, rephrase it to be a standalone question
    const standaloneQuestion = await transformQuery(userProblem);
    //convertting the user query into embeddings
    const { GoogleGenerativeAIEmbeddings } = await import('@langchain/google-genai');
    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        model: 'text-embedding-004',
    });
    // QUERY VECTOR EMBEDDING
    console.log("Generating query embedding...");
    const queryEmbedding = await embeddings.embedQuery(standaloneQuestion);    
    
    // Searching the vector store of PineCone by making connection
    const { Pinecone } = await import('@pinecone-database/pinecone');
    //making connection to Pinecone
    console.log("Connecting to Pinecone...");
    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
    console.log("Searching in Pinecone...");
    const queryResponse = await pineconeIndex.query({ // top 10 documents : 10 metadata -> text part of 10 documents
        vector: queryEmbedding,
        topK: 10,
        includeMetadata: true,
        // namespace: 'dsa-pdf',
    });
    console.log("Query executed successfully!");
    console.log("Top 10 results from Pinecone:");
    queryResponse.matches.forEach((match, index) => {
        console.log(`Result ${index + 1}:`);
        console.log(`ID: ${match.id}`);
        console.log(`Score: ${match.score}`);
        console.log(`Metadata: ${JSON.stringify(match.metadata, null, 2)}`);
    });
    console.log("Query response:", queryResponse);

    //Create the context for the LLM
    const context = queryResponse.matches
                                .map(match => match.metadata.text).join('\n\n---\n\n'); 

    console.log("Context created for the LLM!");

    //Gemini LLM
   
History.push({
    role:'user',
    parts:[{text:userProblem}]
    })              
    const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You have to behave like a Data Structure and Algorithm Expert.
    You will be given a context of relevant information and a user question.
    Your task is to answer the user's question based ONLY on the provided context.
    If the answer is not in the context, you must say "I could not find the answer in the provided document."
    Keep your answers clear, concise, and educational.
      
      Context: ${context}
      `,
    },
   });
    History.push({
    role:'model',
    parts:[{text:response.text}]
  })

  console.log("\n");
  console.log(response.text);
    }
async function main(){
   const userProblem = readlineSync.question("Ask me anything--> ");
   await chatting(userProblem);
   main();
}


main();