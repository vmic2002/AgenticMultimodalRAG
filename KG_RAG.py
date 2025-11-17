from langchain_core.prompts import PromptTemplate
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

class KG_RAG_Agent:
    def __init__(self, llm, graph, verbose=True):
        self.llm_cypher = llm
        self.graph = graph
        self.verbose = verbose

        self.graph.refresh_schema()
        self.schema = self.graph.schema
        if verbose:
            print("Using the following KG schema:")
            print("-"*80)
            print(self.schema)
            print("-"*80)
        self.CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to 
        query a graph database.
        Instructions:
        Use only the provided relationship types and properties in the 
        schema. Do not use any other relationship types or properties that 
        are not provided.
        Schema:
        {schema}
        Note: Do not include any explanations or apologies in your responses.
        Do not respond to any questions that might ask anything else than 
        for you to construct a Cypher statement.
        Do not include any text except the generated Cypher statement.
        Always include the document id in the generated Cypher statement.

        Examples: Here are a few examples of generated Cypher 
        statements for particular questions:


        # What is the subject of the document with id 280?
        MATCH (n:Document)-[:Has_Subject]->(m)
        WHERE n.name = 'Doc 280'
        RETURN n.name as DocId, m.name as subject


        # What is the date in document 279?
        MATCH (n:Document)-[:Has_Date]->(m)
        WHERE n.name = 'Doc 279'
        RETURN n.name as DocId, m.name as date

        # P. Carter is a contact on which document?
        MATCH (p:Person)-[:Contact_On]->(n:Document)
        WHERE p.name = 'P. Carter'
        RETURN n.name as DocId
        The question is:
        {question}"""
        CYPHER_GENERATION_PROMPT = PromptTemplate(
            input_variables=["schema", "question"], 
            template=self.CYPHER_GENERATION_TEMPLATE
        )
        


        self.cypherChain = GraphCypherQAChain.from_llm(
            llm=self.llm_cypher,
            graph=self.graph,
            verbose=self.verbose,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
            allow_dangerous_requests=True,
        )
        
    def cypher_chain(self, question: str) -> str:
        response = self.cypherChain.run(question)
        return response

    def my_cypher_chain(self, question: str) -> str:
        # no dependencies, sometimes cypher_chain says "I dont know" for no reason
        prompt = self.CYPHER_GENERATION_TEMPLATE.format(schema=self.schema, question=question)        
        cypher_response = self.llm_cypher.invoke(prompt)
        response = self.graph.query(cypher_response)
        return cypher_response, response

