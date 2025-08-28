import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';

import { RecursiveCharacterTextSplitter} from 'langchain/text_splitter';

import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import * as dotenv from 'dotenv';
dotenv.config();

import {Pinecone} from '@pinecone-database/pinecone';

import { PineconeStore } from '@langchain/pinecone';

async function indexDocument(){
const PDF_PATH = './dsa.pdf';
const pdfLoader = new PDFLoader(PDF_PATH);
const rawDocs = await pdfLoader.load();

console.log("PDF loaded successfully!");

// console.log(rawDocs.length);
// console.log(JSON.stringify(rawDocs, null, 2));


//Chunking the documents

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200

});
const docs = await textSplitter.splitDocuments(rawDocs);
console.log("Documents chunked successfully!");
// console.log(docs.length);
// console.log(JSON.stringify(docs, null, 2));

// vector embedding and store in the vector db



const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'text-embedding-004',
  });

console.log("Embeddings created successfully!");
//Configure database- Pinecone client
// const { PineconeStore } = require('@langchain/pinecone');
// // const pinecone = require('pinecone-client'); alternative -|
// const pinecone = new PineconeStore();
const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
console.log("Pinecone client configured successfully!");

//langchain (chunking) docs to vector store, embedding and upsert in which database
// const vectorStore = await pineconeIndex.upsert({
//     vectors: await embeddings.embedDocuments(docs.map(doc => doc.pageContent)),
//     namespace: 'dsa-pdf',

// });

await PineconeStore.fromDocuments(docs, embeddings, {
    pineconeIndex: pineconeIndex,
    maxConcurrency: 5,
    // namespace: 'dsa-pdf',
    // textKey: 'pageContent',
  });
console.log("Documents indexed successfully in Pinecone!");
}
indexDocument();