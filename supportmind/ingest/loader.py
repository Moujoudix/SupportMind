"""
Data Ingestion Module.
Loads CSV data and populates database + vector store.
"""

import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from supportmind.config.settings import get_config, Paths
from supportmind.models.schemas import Document
from supportmind.stores.database import Database
from supportmind.stores.vector_store import VectorStore


class DataIngester:
    """
    GPU-optimized data ingestion.
    Loads CSVs and indexes all documents.
    """

    def __init__(
        self,
        db: Database = None,
        vector_store: VectorStore = None,
        paths: Paths = None
    ):
        """
        Initialize data ingester.

        Args:
            db: Database instance (created if None)
            vector_store: VectorStore instance (created if None)
            paths: Paths configuration (uses config default if None)
        """
        config = get_config()
        self.db = db or Database()
        self.vs = vector_store or VectorStore()
        self.paths = paths or config.paths
        self.dfs: Dict[str, pd.DataFrame] = {}
        self.stats: Dict[str, int] = {}

    def load_csvs(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files."""
        print("\n" + "=" * 60)
        print("LOADING CSV FILES")
        print("=" * 60)

        files = {
            'conversations': self.paths.conversations,
            'existing_kb': self.paths.existing_kb,
            'knowledge_articles': self.paths.knowledge_articles,
            'scripts_master': self.paths.scripts_master,
            'questions': self.paths.questions,
            'tickets': self.paths.tickets,
            'kb_lineage': self.paths.kb_lineage,
            'learning_events': self.paths.learning_events,
        }

        for name, path in files.items():
            try:
                path = Path(path)
                if path.exists():
                    df = pd.read_csv(path)
                    self.dfs[name] = df
                    self.stats[f'loaded_{name}'] = len(df)
                    print(f"✅ {name}: {len(df)} rows")
                else:
                    print(f"⚠️ Not found: {path}")
                    self.dfs[name] = pd.DataFrame()
            except Exception as e:
                print(f"❌ Error {name}: {e}")
                self.dfs[name] = pd.DataFrame()

        return self.dfs

    def ingest_knowledge_articles(self) -> int:
        """
        Ingest KB articles from Knowledge_Articles and Existing_Knowledge_Articles.
        """
        print("\n📚 Ingesting Knowledge Articles...")
        documents = []
        seen_ids = set()

        # Knowledge_Articles
        df = self.dfs.get('knowledge_articles', pd.DataFrame())
        if not df.empty:
            for _, row in df.iterrows():
                kb_id = str(row['KB_Article_ID'])
                if kb_id in seen_ids or pd.isna(row['KB_Article_ID']):
                    continue
                seen_ids.add(kb_id)

                doc = Document(
                    doc_id=f"KB_{kb_id}",
                    doc_type="KB",
                    source_id=kb_id,
                    title=str(row['Title']),
                    content=str(row['Body']),
                    metadata={
                        'tags': str(row['Tags']) if pd.notna(row.get('Tags')) else '',
                        'module': str(row['Module']) if pd.notna(row.get('Module')) else '',
                        'category': str(row['Category']) if pd.notna(row.get('Category')) else '',
                        'status': str(row['Status']),
                        'source_type': str(row['Source_Type']),
                        'source': 'knowledge_articles'
                    },
                    created_at=str(row['Created_At']) if pd.notna(row.get('Created_At')) else '',
                    updated_at=str(row['Updated_At']) if pd.notna(row.get('Updated_At')) else ''
                )

                self._store_document(doc)
                documents.append(doc)

        # Existing_Knowledge_Articles
        df_existing = self.dfs.get('existing_kb', pd.DataFrame())
        if not df_existing.empty:
            for _, row in df_existing.iterrows():
                kb_id = str(row['KB_Article_ID'])
                if kb_id in seen_ids or pd.isna(row['KB_Article_ID']):
                    continue
                seen_ids.add(kb_id)

                doc = Document(
                    doc_id=f"KB_{kb_id}",
                    doc_type="KB",
                    source_id=kb_id,
                    title=str(row['Title']),
                    content=str(row['Body']),
                    metadata={
                        'product': str(row['Product']),
                        'experience': str(row['Experience']),
                        'url': str(row['URL']),
                        'source_table': str(row['Source_Table']),
                        'source_type': str(row['Source_Type']),
                        'source': 'existing_kb'
                    }
                )

                self._store_document(doc)
                documents.append(doc)

        # Add to vector store
        if documents:
            self.vs.add_documents(documents, show_progress=True)

        self.stats['kb_articles'] = len(documents)
        print(f"   ✅ {len(documents)} KB articles indexed")
        return len(documents)

    def ingest_scripts(self) -> int:
        """Ingest scripts from Scripts_Master."""
        print("\n📜 Ingesting Scripts...")
        df = self.dfs.get('scripts_master', pd.DataFrame())
        if df.empty:
            print("   ⚠️ No scripts data")
            return 0

        documents = []
        for _, row in df.iterrows():
            script_id = str(row['Script_ID'])
            if pd.isna(row['Script_ID']):
                continue

            content = f"""Purpose: {row['Script_Purpose']}

Inputs Required: {row['Script_Inputs']}

Script:
{row['Script_Text_Sanitized']}"""

            doc = Document(
                doc_id=f"SCRIPT_{script_id}",
                doc_type="SCRIPT",
                source_id=script_id,
                title=str(row['Script_Title']),
                content=content,
                metadata={
                    'module': str(row['Module']),
                    'category': str(row['Category']),
                    'source': str(row['Source']),
                    'purpose': str(row['Script_Purpose']),
                    'inputs': str(row['Script_Inputs'])
                }
            )

            self._store_document(doc)
            documents.append(doc)

        if documents:
            self.vs.add_documents(documents, show_progress=True)

        self.stats['scripts'] = len(documents)
        print(f"   ✅ {len(documents)} scripts indexed")
        return len(documents)

    def ingest_tickets(self) -> int:
        """Ingest tickets from Tickets CSV."""
        print("\n🎫 Ingesting Tickets...")
        df = self.dfs.get('tickets', pd.DataFrame())
        if df.empty:
            print("   ⚠️ No tickets data")
            return 0

        documents = []
        for _, row in df.iterrows():
            ticket_num = str(row['Ticket_Number'])
            if pd.isna(row['Ticket_Number']):
                continue

            # Store raw ticket
            self.db.insert("tickets", {
                'Ticket_Number': ticket_num,
                'Conversation_ID': str(row['Conversation_ID']),
                'Created_At': str(row['Created_At']),
                'Closed_At': str(row['Closed_At']),
                'Status': str(row['Status']),
                'Priority': str(row['Priority']),
                'Tier': int(row['Tier']),
                'Product': str(row['Product']),
                'Module': str(row['Module']),
                'Category': str(row['Category']),
                'Case_Type': str(row['Case_Type']),
                'Account_Name': str(row['Account_Name']),
                'Property_Name': str(row['Property_Name']),
                'Property_City': str(row['Property_City']),
                'Property_State': str(row['Property_State']),
                'Contact_Name': str(row['Contact_Name']),
                'Contact_Role': str(row['Contact_Role']),
                'Contact_Email': str(row['Contact_Email']),
                'Contact_Phone': str(row['Contact_Phone']),
                'Subject': str(row['Subject']),
                'Description': str(row['Description']),
                'Resolution': str(row['Resolution']),
                'Root_Cause': str(row['Root_Cause']),
                'Tags': str(row['Tags']),
                'KB_Article_ID': str(row['KB_Article_ID']) if pd.notna(row.get('KB_Article_ID')) else '',
                'Script_ID': str(row['Script_ID']) if pd.notna(row.get('Script_ID')) else '',
                'Generated_KB_Article_ID': str(row['Generated_KB_Article_ID']) if pd.notna(row.get('Generated_KB_Article_ID')) else ''
            })

            # Create searchable document
            content = f"""Subject: {row['Subject']}

Description: {row['Description']}

Resolution: {row['Resolution']}

Root Cause: {row['Root_Cause']}"""

            doc = Document(
                doc_id=f"TICKET_{ticket_num}",
                doc_type="TICKET",
                source_id=ticket_num,
                title=str(row['Subject']),
                content=content,
                metadata={
                    'product': str(row['Product']),
                    'module': str(row['Module']),
                    'category': str(row['Category']),
                    'tier': int(row['Tier']),
                    'status': str(row['Status']),
                    'priority': str(row['Priority']),
                    'case_type': str(row['Case_Type']),
                    'tags': str(row['Tags']),
                    'kb_article_id': str(row['KB_Article_ID']) if pd.notna(row.get('KB_Article_ID')) else '',
                    'script_id': str(row['Script_ID']) if pd.notna(row.get('Script_ID')) else ''
                },
                created_at=str(row['Created_At'])
            )

            self._store_document(doc)
            documents.append(doc)

        if documents:
            self.vs.add_documents(documents, show_progress=True)

        self.stats['tickets'] = len(documents)
        print(f"   ✅ {len(documents)} tickets indexed")
        return len(documents)

    def ingest_conversations(self) -> int:
        """Ingest conversations from Conversations CSV."""
        print("\n💬 Ingesting Conversations...")
        df = self.dfs.get('conversations', pd.DataFrame())
        if df.empty:
            print("   ⚠️ No conversations data")
            return 0

        documents = []
        for _, row in df.iterrows():
            conv_id = str(row['Conversation_ID'])
            if pd.isna(row['Conversation_ID']):
                continue

            # Store raw conversation
            self.db.insert("conversations", {
                'Conversation_ID': conv_id,
                'Ticket_Number': str(row['Ticket_Number']),
                'Channel': str(row['Channel']),
                'Conversation_Start': str(row['Conversation_Start']),
                'Conversation_End': str(row['Conversation_End']),
                'Customer_Role': str(row['Customer_Role']),
                'Agent_Name': str(row['Agent_Name']),
                'Product': str(row['Product']),
                'Category': str(row['Category']),
                'Issue_Summary': str(row['Issue_Summary']),
                'Transcript': str(row['Transcript']),
                'Sentiment': str(row['Sentiment'])
            })

            content = f"""Issue Summary: {row['Issue_Summary']}

Transcript:
{row['Transcript']}"""

            doc = Document(
                doc_id=f"CONV_{conv_id}",
                doc_type="CONVERSATION",
                source_id=conv_id,
                title=str(row['Issue_Summary']),
                content=content,
                metadata={
                    'ticket_number': str(row['Ticket_Number']),
                    'channel': str(row['Channel']),
                    'product': str(row['Product']),
                    'category': str(row['Category']),
                    'agent': str(row['Agent_Name']),
                    'sentiment': str(row['Sentiment'])
                },
                created_at=str(row['Conversation_Start'])
            )

            self._store_document(doc)
            documents.append(doc)

        if documents:
            self.vs.add_documents(documents, show_progress=True)

        self.stats['conversations'] = len(documents)
        print(f"   ✅ {len(documents)} conversations indexed")
        return len(documents)

    def ingest_questions(self) -> int:
        """Ingest questions for evaluation."""
        print("\n❓ Ingesting Questions...")
        df = self.dfs.get('questions', pd.DataFrame())
        if df.empty:
            print("   ⚠️ No questions data")
            return 0

        count = 0
        for _, row in df.iterrows():
            self.db.insert("questions", {
                'Question_ID': str(row['Question_ID']),
                'Source': str(row['Source']),
                'Product': str(row['Product']),
                'Category': str(row['Category']),
                'Module': str(row['Module']),
                'Difficulty': str(row['Difficulty']),
                'Question_Text': str(row['Question_Text']),
                'Answer_Type': str(row['Answer_Type']),
                'Target_ID': str(row['Target_ID']),
                'Target_Title': str(row['Target_Title'])
            })
            count += 1

        self.stats['questions'] = count
        print(f"   ✅ {count} questions loaded")
        return count

    def ingest_kb_lineage(self) -> int:
        """Ingest KB lineage records."""
        print("\n🔗 Ingesting KB Lineage...")
        df = self.dfs.get('kb_lineage', pd.DataFrame())
        if df.empty:
            print("   ⚠️ No lineage data")
            return 0

        count = 0
        for _, row in df.iterrows():
            self.db.insert("kb_lineage", {
                'KB_Article_ID': str(row['KB_Article_ID']),
                'Source_Type': str(row['Source_Type']),
                'Source_ID': str(row['Source_ID']),
                'Relationship': str(row['Relationship']),
                'Evidence_Snippet': str(row['Evidence_Snippet']),
                'Event_Timestamp': str(row['Event_Timestamp'])
            })
            count += 1

        self.stats['kb_lineage'] = count
        print(f"   ✅ {count} lineage records loaded")
        return count

    def ingest_learning_events(self) -> int:
        """Ingest learning events."""
        print("\n📖 Ingesting Learning Events...")
        df = self.dfs.get('learning_events', pd.DataFrame())
        if df.empty:
            print("   ⚠️ No learning events data")
            return 0

        count = 0
        for _, row in df.iterrows():
            self.db.insert("learning_events", {
                'Event_ID': str(row['Event_ID']),
                'Trigger_Ticket_Number': str(row['Trigger_Ticket_Number']),
                'Trigger_Conversation_ID': str(row['Trigger_Conversation_ID']),
                'Detected_Gap': str(row['Detected_Gap']),
                'Proposed_KB_Article_ID': str(row['Proposed_KB_Article_ID']),
                'Draft_Summary': str(row['Draft_Summary']),
                'Final_Status': str(row['Final_Status']),
                'Reviewer_Role': str(row['Reviewer_Role']),
                'Event_Timestamp': str(row['Event_Timestamp'])
            })
            count += 1

        self.stats['learning_events'] = count
        print(f"   ✅ {count} learning events loaded")
        return count

    def _store_document(self, doc: Document):
        """Store document in database."""
        self.db.insert("documents", {
            'doc_id': doc.doc_id,
            'doc_type': doc.doc_type,
            'source_id': doc.source_id,
            'title': doc.title,
            'content': doc.content,
            'metadata': doc.metadata,
            'version': doc.version,
            'created_at': doc.created_at,
            'updated_at': doc.updated_at,
            'content_hash': doc.content_hash(),
            'is_active': 1
        })

    def run_full_ingestion(self) -> Dict[str, int]:
        """Run complete ingestion pipeline."""
        import time

        print("\n" + "=" * 60)
        print("FULL DATA INGESTION PIPELINE")
        print("=" * 60)

        start_time = time.time()

        self.load_csvs()
        self.ingest_knowledge_articles()
        self.ingest_scripts()
        self.ingest_tickets()
        self.ingest_conversations()
        self.ingest_questions()
        self.ingest_kb_lineage()
        self.ingest_learning_events()

        # Save index
        self.vs.save()

        elapsed = time.time() - start_time

        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        print(f"Total time: {elapsed:.1f}s")
        print(f"Vector store: {self.vs.count()} documents")
        print(f"By type: {self.vs.counts_by_type()}")

        return self.stats
