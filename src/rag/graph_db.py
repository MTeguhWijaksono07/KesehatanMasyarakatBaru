from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI', '').strip("'\"")
NEO4J_USER = os.getenv('NEO4J_USER', '').strip("'\"")
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', '').strip("'\"")

class Neo4jHandler:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def query(self, cypher, parameters=None):
        with self.driver.session() as session:
            result = session.run(cypher, parameters)
            return [dict(record) for record in result]

    def get_all_chunks(self):
        cypher = """
            MATCH (c:Chunk)
            RETURN c.chunk_id AS chunk_id, c.content AS content,
                   c.page AS page, c.chapter AS chapter,
                   c.source AS source, c.embedding AS embedding
        """
        return self.query(cypher)

    def check_connection(self):
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception:
            return False

neo4j_handler = Neo4jHandler()
