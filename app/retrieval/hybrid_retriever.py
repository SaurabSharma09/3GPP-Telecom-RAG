from pathlib import Path
import pickle
import re
import sys


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from app.retrieval.query_analyzer import QueryAnalyzer


VECTOR_DIR = Path("data/processed/vector_store")

MODEL_NAME = "sentence-transformers/" "all-MiniLM-L6-v2"


class HybridRetriever:

    # ======================================================
    # QUERY INTENT TERMS
    # ======================================================

    QUERY_TYPE_TERMS = {
        "ROLE": [
            "function",
            "functions",
            "functionality",
            "responsible",
            "responsibility",
            "responsibilities",
            "manage",
            "manages",
            "shall support",
        ],
        "INTERFACE": ["interface", "reference point", "connects", "connection"],
        "DEFINITION": ["defined", "definition", "refers to", "means", "is defined as"],
        "PROCEDURE": [
            "procedure",
            "step",
            "steps",
            "request",
            "response",
            "initiates",
            "initiated",
            "establishment",
            "release",
            "modification",
        ],
        "COMPARISON": [
            "difference",
            "different",
            "compared",
            "whereas",
            "respectively",
            "in contrast",
        ],
        "GENERAL": [],
    }

    # ======================================================
    # CONCEPT PHRASES
    # ======================================================

    CONCEPT_PATTERNS = {
        "REGISTRATION": [
            "ue registration",
            "registration management",
            "registration procedure",
            "registration",
        ],
        "PDU_SESSION_ESTABLISHMENT": [
            "pdu session establishment",
            "pdu session establishment procedure",
            "establishment of a pdu session",
            "establish a pdu session",
        ],
        "SERVICE_REQUEST": ["service request", "service request procedure"],
        "DEREGISTRATION": [
            "deregistration",
            "de-registration",
            "deregistration procedure",
        ],
        "HANDOVER": ["handover", "handover procedure", "xn handover", "n2 handover"],
        "PAGING": ["paging", "paging procedure"],
        "AUTHENTICATION": [
            "authentication",
            "authentication procedure",
            "primary authentication",
        ],
    }

    # ======================================================
    # INITIALIZATION
    # ======================================================

    def __init__(
        self, vector_k: int = 40, bm25_k: int = 40, final_k: int = 15, rrf_k: int = 60
    ):
        """
        Hybrid retrieval pipeline:

        1. FAISS semantic retrieval
        2. BM25 lexical retrieval
        3. Reciprocal Rank Fusion
        4. Multi-entity matching
        5. Concept matching
        6. Query-type-aware ranking
        7. Section-title-aware ranking
        8. Interface-specific ranking
        9. Exact interface entity ranking
        """

        self.vector_k = vector_k
        self.bm25_k = bm25_k
        self.final_k = final_k
        self.rrf_k = rrf_k

        # ==================================================
        # QUERY ANALYZER
        # ==================================================

        print("Loading query analyzer...")

        self.query_analyzer = QueryAnalyzer()

        # ==================================================
        # FAISS
        # ==================================================

        print("Loading FAISS index...")

        self.vector_index = faiss.read_index(str(VECTOR_DIR / "3gpp.index"))

        with open(VECTOR_DIR / "chunks.pkl", "rb") as file:

            self.vector_chunks = pickle.load(file)

        # ==================================================
        # BM25
        # ==================================================

        print("Loading BM25 index...")

        with open(VECTOR_DIR / "bm25.pkl", "rb") as file:

            self.bm25 = pickle.load(file)

        with open(VECTOR_DIR / "bm25_chunks.pkl", "rb") as file:

            self.bm25_chunks = pickle.load(file)

        # ==================================================
        # EMBEDDING MODEL
        # ==================================================

        print("Loading embedding model...")

        self.embedding_model = SentenceTransformer(MODEL_NAME)

        print("Hybrid retriever ready.")

    # ======================================================
    # TOKENIZATION
    # ======================================================

    @staticmethod
    def tokenize(text: str):

        text = text.lower()

        return re.findall(r"[a-zA-Z0-9]+(?:[-_/][a-zA-Z0-9]+)*", text)

    # ======================================================
    # EXACT ENTITY OCCURRENCES
    # ======================================================

    @staticmethod
    def entity_occurrences(text: str, entity: str):

        if not entity:
            return 0

        pattern = r"\b" + re.escape(entity.lower()) + r"\b"

        return len(re.findall(pattern, text.lower()))

    # ======================================================
    # ENTITY COVERAGE
    # ======================================================

    @staticmethod
    def entity_coverage(text: str, entities):

        if not entities:
            return 0.0

        matched = 0

        for entity in entities:

            if HybridRetriever.entity_occurrences(text, entity) > 0:

                matched += 1

        return matched / len(entities)

    # ======================================================
    # QUERY TYPE SCORE
    # ======================================================

    def query_type_score(self, text: str, query_type: str):

        terms = self.QUERY_TYPE_TERMS.get(query_type, [])

        if not terms:
            return 0.0

        text_lower = text.lower()

        matches = 0

        for term in terms:

            if term.lower() in text_lower:

                matches += 1

        return matches / len(terms)

    # ======================================================
    # CONCEPT SCORE
    # ======================================================

    @staticmethod
    def concept_score(query: str, text: str, section_title: str):

        query_lower = query.lower()

        text_lower = text.lower()

        title_lower = section_title.lower()

        best_score = 0.0

        for phrases in HybridRetriever.CONCEPT_PATTERNS.values():

            for phrase in phrases:

                if phrase not in query_lower:
                    continue

                # Exact concept in section title
                if phrase in title_lower:

                    best_score = max(best_score, 1.0)

                    continue

                # Exact concept in chunk
                if phrase in text_lower:

                    best_score = max(best_score, 0.75)

        return best_score

    # ======================================================
    # SECTION TITLE RELEVANCE
    # ======================================================

    @staticmethod
    def section_score(query: str, query_type: str, entities, section_title: str):

        if not section_title:
            return 0.0

        title = section_title.lower().strip()

        query_lower = query.lower().strip()

        score = 0.0

        # ==================================================
        # ENTITY PRESENCE IN SECTION TITLE
        # ==================================================

        if entities:

            matched_entities = 0

            for entity in entities:

                entity_lower = entity.lower().strip()

                if not entity_lower:
                    continue

                pattern = r"\b" + re.escape(entity_lower) + r"\b"

                if re.search(pattern, title):

                    matched_entities += 1

            entity_coverage = matched_entities / len(entities)

            score += 0.70 * entity_coverage

        # ==================================================
        # QUERY TYPE + SECTION TITLE
        # ==================================================

        if query_type == "ROLE":

            role_title_terms = [
                "function",
                "functions",
                "functionality",
                "role",
                "general",
                "description",
            ]

            if any(term in title for term in role_title_terms):

                score += 0.15

        elif query_type == "INTERFACE":

            interface_title_terms = [
                "interface",
                "reference point",
                "reference points",
                "connectivity",
                "connection",
                "interaction",
            ]

            if any(term in title for term in interface_title_terms):

                score += 0.20

        elif query_type == "PROCEDURE":

            procedure_title_terms = [
                "procedure",
                "establishment",
                "release",
                "modification",
                "registration",
                "service request",
            ]

            if any(term in title for term in procedure_title_terms):

                score += 0.20

        elif query_type == "COMPARISON":

            comparison_title_terms = [
                "interaction",
                "comparison",
                "difference",
                "relationship",
                "architecture",
            ]

            if any(term in title for term in comparison_title_terms):

                score += 0.15

        elif query_type == "DEFINITION":

            definition_title_terms = [
                "definition",
                "definitions",
                "terms",
                "abbreviations",
            ]

            if any(term in title for term in definition_title_terms):

                score += 0.20

        # ==================================================
        # EXACT CONCEPT IN SECTION TITLE
        # ==================================================

        for phrases in HybridRetriever.CONCEPT_PATTERNS.values():

            for phrase in phrases:

                if phrase in query_lower:

                    if phrase in title:

                        score += 0.30

        return min(score, 1.0)

    # ======================================================
    # CANONICAL CONCEPT SECTION RELEVANCE
    # ======================================================

    @staticmethod
    def canonical_concept_score(
        query: str,
        query_type: str,
        entities,
        section_number: str,
        section_title: str,
        section_type: str,
    ):
        """
        Prefer primary/canonical clauses for broad telecom questions.

        Canonical mappings:
            - PDU Session Establishment
            - AMF role
            - SMF role
            - UPF role
        """
        query_lower = (query or "").lower().strip()
        number = (section_number or "").lower().strip()
        title = (section_title or "").lower().strip()
        normalized_type = (section_type or "").lower().strip()

        if normalized_type == "cover":
            return 0.0

        score = 0.0

        # Canonical network-function role sections.
        if query_type == "ROLE":
            role_sections = {
                "amf": "6.2.1",
                "smf": "6.2.2",
                "upf": "6.2.3",
            }

            normalized_entities = {
                str(entity).lower().strip() for entity in (entities or [])
            }

            for entity, canonical_section in role_sections.items():
                present = entity in normalized_entities or re.search(
                    rf"\b{re.escape(entity)}\b",
                    query_lower,
                )

                if present and number == canonical_section:
                    score = max(score, 1.0)

        # Canonical PDU Session Establishment clauses.
        if (
            "pdu session establishment" in query_lower
            or "establishment of a pdu session" in query_lower
            or "establish a pdu session" in query_lower
        ):
            if number in {"4.3.2.1", "4.3.2.2.1"}:
                score = max(score, 1.0)

            if "pdu session establishment" in title and "general" in title:
                score = max(score, 1.0)

            secondary_terms = [
                "secondary authorization",
                "secondary authentication",
                "dn-aaa",
                "i-smf",
                "v-smf",
                "context transfer",
                "enhancement",
                "impact",
            ]

            if any(term in title for term in secondary_terms):
                score = min(score, 0.10)

        # Canonical Registration procedure clauses.
        if "registration" in query_lower and "deregistration" not in query_lower:
            # TS 23.502 §4.2.2.2.2 is the primary registration procedure.
            if number in {"4.2.2.2", "4.2.2.2.2"}:
                score = max(score, 1.0)

            # TS 23.501 §5.3.2 Registration Management is the architecture.
            if number in {"5.3.2", "5.3.2.1"}:
                score = max(score, 0.80)

            if "registration" in title and "general" in title:
                score = max(score, 0.80)

        return min(score, 1.0)

    # ======================================================
    # INTERFACE-SPECIFIC RELEVANCE
    # ======================================================

    @staticmethod
    def interface_score(text: str, section_title: str, entities):
        """
        General interface relevance.

        Rewards:

        - exact interface entity
        - interface terminology
        - reference-point terminology
        - AMF + SMF relationship
        - interface-oriented section titles
        """

        if not entities:
            return 0.0

        text_lower = text.lower()

        title_lower = (section_title or "").lower()

        score = 0.0

        interface_terms = [
            "interface",
            "reference point",
            "reference points",
            "interface between",
            "interface with",
            "connects",
            "connection between",
            "interaction between",
            "communication between",
        ]

        relationship_phrases = [
            "between the amf and smf",
            "between amf and smf",
            "amf and smf",
            "amf to smf",
            "smf to amf",
            "amf communicates with smf",
            "smf communicates with amf",
        ]

        interface_title_terms = [
            "interface",
            "reference point",
            "reference points",
            "interaction",
            "connectivity",
            "connection",
        ]

        for entity in entities:

            entity_lower = entity.lower().strip()

            if not entity_lower:
                continue

            entity_pattern = r"\b" + re.escape(entity_lower) + r"\b"

            if not re.search(entity_pattern, text_lower):

                continue

            # Exact entity occurrence
            score += 0.20

            # Entity in title
            if re.search(entity_pattern, title_lower):

                score += 0.30

            # Interface terminology
            matched_interface_terms = sum(
                1 for term in interface_terms if term in text_lower
            )

            score += min(matched_interface_terms * 0.10, 0.30)

            # AMF + SMF relationship
            has_amf = bool(re.search(r"\bAMF\b", text))

            has_smf = bool(re.search(r"\bSMF\b", text))

            if has_amf and has_smf:

                score += 0.25

            # Relationship phrase
            if any(phrase in text_lower for phrase in relationship_phrases):

                score += 0.20

            # Interface-oriented title
            if any(term in title_lower for term in interface_title_terms):

                score += 0.20

        return min(score, 1.0)

    # ======================================================
    # EXACT INTERFACE ENTITY SCORE
    # ======================================================

    @staticmethod
    def exact_interface_entity_score(text: str, section_title: str, entities):
        """
        Strong signal for queries such as:

            What is the N11 interface?

        Merely mentioning N11 is not enough.

        Stronger evidence:

            N11 + interface
            N11 + reference point
            N11 interface
            N11 reference point
            interface between relevant network functions
        """

        if not entities:
            return 0.0

        text_lower = text.lower()

        title_lower = (section_title or "").lower()

        score = 0.0

        for entity in entities:

            entity_lower = entity.lower().strip()

            if not entity_lower:
                continue

            pattern = r"\b" + re.escape(entity_lower) + r"\b"

            matches = list(re.finditer(pattern, text_lower))

            if not matches:
                continue

            # Exact entity exists.
            score += 0.30

            # Entity in section title.
            if re.search(pattern, title_lower):

                score += 0.30

            # Look around each entity occurrence.
            for match in matches:

                start = max(0, match.start() - 180)

                end = min(len(text_lower), match.end() + 180)

                context = text_lower[start:end]

                # Direct interface terminology.
                if "interface" in context:

                    score += 0.20

                # Reference point terminology.
                if "reference point" in context:

                    score += 0.20

                # Very strong exact phrases.
                strong_terms = [
                    "n11 interface",
                    "n11 reference point",
                    "reference point n11",
                    "interface n11",
                ]

                if any(term in context for term in strong_terms):

                    score += 0.30

        return min(score, 1.0)

    # ======================================================
    # ADVANCED RELEVANCE FEATURES
    # ======================================================

    def calculate_relevance_features(self, result, query, query_type, entities):

        chunk = result["chunk"]

        text = chunk.get("text", "")

        metadata = chunk.get("metadata", {})

        section_title = metadata.get("section_title", "") or ""

        section_number = metadata.get("section_number", "") or ""
        section_type = metadata.get("section_type", "") or ""

        text_lower = text.lower()

        title_lower = section_title.lower()

        # ==================================================
        # ENTITY COVERAGE
        # ==================================================

        coverage = self.entity_coverage(text, entities)

        # ==================================================
        # ENTITY SCORE
        # ==================================================

        entity_score = 0.0

        if entities:

            individual_scores = []

            for entity in entities:

                entity_lower = entity.lower()

                occurrences = self.entity_occurrences(text, entity)

                pattern = r"\b" + re.escape(entity_lower) + r"\b"

                title_match = bool(re.search(pattern, title_lower))

                beginning_match = bool(re.search(pattern, text_lower[:500]))

                score = 0.0

                # Entity in section title
                if title_match:

                    score += 0.70

                elif occurrences > 0:

                    score += 0.40

                # Repeated entity
                if occurrences >= 2:

                    score += 0.15

                if occurrences >= 4:

                    score += 0.10

                # Entity near beginning
                if beginning_match:

                    score += 0.10

                score = min(score, 1.0)

                individual_scores.append(score)

            entity_score = sum(individual_scores) / len(individual_scores)

            # Multi-entity queries
            if len(entities) > 1:

                entity_score = entity_score * (0.5 + 0.5 * coverage)

        # ==================================================
        # INTENT SCORE
        # ==================================================

        intent_score = self.query_type_score(text, query_type)

        # ==================================================
        # ROLE
        # ==================================================

        if query_type == "ROLE":

            role_phrases = [
                "is responsible for",
                "are responsible for",
                "includes the following functionality",
                "following functionality",
                "shall support",
                "provides the following",
                "functions provided by",
                "functionality of",
            ]

            matches = sum(1 for phrase in role_phrases if phrase in text_lower)

            if matches:

                intent_score += min(matches * 0.20, 0.50)

        # ==================================================
        # INTERFACE
        # ==================================================

        elif query_type == "INTERFACE":

            interface_phrases = [
                "interface between",
                "interface with",
                "reference point",
                "interface is",
                "interface supports",
                "connects",
                "connection between",
            ]

            matches = sum(1 for phrase in interface_phrases if phrase in text_lower)

            if matches:

                intent_score += min(matches * 0.20, 0.50)

            if entities:

                title_entities = sum(
                    1 for entity in entities if entity.lower() in title_lower
                )

                if title_entities:

                    intent_score += min(title_entities * 0.25, 0.50)

        # ==================================================
        # DEFINITION
        # ==================================================

        elif query_type == "DEFINITION":

            definition_phrases = [
                "is defined as",
                "is defined",
                "refers to",
                "means",
                "defined in",
            ]

            matches = sum(1 for phrase in definition_phrases if phrase in text_lower)

            if matches:

                intent_score += min(matches * 0.20, 0.50)

        # ==================================================
        # PROCEDURE
        # ==================================================

        elif query_type == "PROCEDURE":

            procedure_phrases = [
                "procedure",
                "step",
                "steps",
                "request",
                "response",
                "initiates",
                "shall send",
                "shall receive",
                "establishment",
            ]

            matches = sum(1 for phrase in procedure_phrases if phrase in text_lower)

            if matches:

                intent_score += min(matches * 0.10, 0.50)

        # ==================================================
        # COMPARISON
        # ==================================================

        elif query_type == "COMPARISON":

            comparison_phrases = [
                "difference",
                "whereas",
                "respectively",
                "compared to",
                "in contrast",
                "separate network functions",
                "different network functions",
            ]

            matches = sum(1 for phrase in comparison_phrases if phrase in text_lower)

            if matches:

                intent_score += min(matches * 0.15, 0.50)

            # Both entities are important.
            if len(entities) >= 2 and coverage == 1.0:

                intent_score += 0.30

        # ==================================================
        # CONCEPT
        # ==================================================

        concept = self.concept_score(query, text, section_title)

        # ==================================================
        # SECTION TITLE
        # ==================================================

        section = self.section_score(
            query=query,
            query_type=query_type,
            entities=entities,
            section_title=section_title,
        )
        if section_type == "cover":
            section = 0.0

        canonical_concept = self.canonical_concept_score(
            query=query,
            query_type=query_type,
            entities=entities,
            section_number=section_number,
            section_title=section_title,
            section_type=section_type,
        )

        # ==================================================
        # INTERFACE-SPECIFIC SCORE
        # ==================================================

        interface = 0.0

        exact_interface = 0.0

        if query_type == "INTERFACE":

            interface = self.interface_score(
                text=text, section_title=section_title, entities=entities
            )

            exact_interface = self.exact_interface_entity_score(
                text=text, section_title=section_title, entities=entities
            )

        # ==================================================
        # NORMALIZE INTENT
        # ==================================================

        intent_score = min(intent_score, 1.0)

        return {
            "entity_score": entity_score,
            "entity_coverage": coverage,
            "intent_score": intent_score,
            "concept_score": concept,
            "section_score": section,
            "canonical_concept_score": canonical_concept,
            "section_type": section_type,
            "interface_score": interface,
            "exact_interface_score": exact_interface,
        }

    # ======================================================
    # VECTOR SEARCH
    # ======================================================

    def vector_search(self, query: str):

        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        )

        query_embedding = np.asarray(query_embedding, dtype="float32")

        scores, indices = self.vector_index.search(query_embedding, self.vector_k)

        results = []

        for rank, (score, index) in enumerate(zip(scores[0], indices[0]), start=1):

            if index < 0:
                continue

            results.append(
                {
                    "index": int(index),
                    "rank": rank,
                    "score": float(score),
                    "chunk": self.vector_chunks[index],
                }
            )

        return results

    # ======================================================
    # BM25 SEARCH
    # ======================================================

    def bm25_search(self, query: str):

        tokens = self.tokenize(query)

        scores = self.bm25.get_scores(tokens)

        top_indices = np.argsort(scores)[::-1][: self.bm25_k]

        results = []

        for rank, index in enumerate(top_indices, start=1):

            results.append(
                {
                    "index": int(index),
                    "rank": rank,
                    "score": float(scores[index]),
                    "chunk": self.bm25_chunks[index],
                }
            )

        return results

    # ======================================================
    # RECIPROCAL RANK FUSION
    # ======================================================

    def reciprocal_rank_fusion(self, vector_results, bm25_results):

        combined = {}

        # ==================================================
        # FAISS
        # ==================================================

        for result in vector_results:

            index = result["index"]

            if index not in combined:

                combined[index] = {
                    "chunk": result["chunk"],
                    "vector_rank": None,
                    "bm25_rank": None,
                    "vector_score": None,
                    "bm25_score": None,
                    "rrf_score": 0.0,
                }

            combined[index]["vector_rank"] = result["rank"]

            combined[index]["vector_score"] = result["score"]

            combined[index]["rrf_score"] += 1.0 / (self.rrf_k + result["rank"])

        # ==================================================
        # BM25
        # ==================================================

        for result in bm25_results:

            index = result["index"]

            if index not in combined:

                combined[index] = {
                    "chunk": result["chunk"],
                    "vector_rank": None,
                    "bm25_rank": None,
                    "vector_score": None,
                    "bm25_score": None,
                    "rrf_score": 0.0,
                }

            combined[index]["bm25_rank"] = result["rank"]

            combined[index]["bm25_score"] = result["score"]

            combined[index]["rrf_score"] += 1.0 / (self.rrf_k + result["rank"])

        # ==================================================
        # LIST
        # ==================================================

        results = []

        for index, item in combined.items():

            results.append(
                {
                    "index": index,
                    "rrf_score": item["rrf_score"],
                    "vector_rank": item["vector_rank"],
                    "bm25_rank": item["bm25_rank"],
                    "vector_score": item["vector_score"],
                    "bm25_score": item["bm25_score"],
                    "chunk": item["chunk"],
                }
            )

        return results

    # ======================================================
    # QUERY-AWARE RANKING
    # ======================================================

    def apply_query_intent(self, results, query, query_type, entities):
        """
        Final first-stage retrieval ranking.

        RRF:
            Base semantic + lexical retrieval

        Entity:
            Entity relevance

        Coverage:
            Whether all query entities are represented

        Intent:
            Role/interface/procedure/etc.

        Concept:
            Exact multi-word concept match

        Section:
            Section-title relevance

        Interface:
            General interface relevance

        Exact Interface:
            Strong exact interface/entity/context match
        """

        ENTITY_WEIGHT = 0.080

        COVERAGE_WEIGHT = 0.055

        CONCEPT_WEIGHT = 0.070

        SECTION_WEIGHT = 0.050

        # Strong signal for the primary clause of a broad concept.
        CANONICAL_CONCEPT_WEIGHT = 0.120

        INTENT_WEIGHT = 0.035

        # Stronger than generic interface relevance.
        INTERFACE_WEIGHT = 0.045

        # Strongest special signal for interface entities.
        EXACT_INTERFACE_WEIGHT = 0.065

        for result in results:

            features = self.calculate_relevance_features(
                result, query, query_type, entities
            )

            entity_score = features["entity_score"]

            coverage = features["entity_coverage"]

            intent_score = features["intent_score"]

            concept_score = features["concept_score"]

            section_score = features["section_score"]

            canonical_concept_score = features["canonical_concept_score"]

            interface_score = features["interface_score"]

            exact_interface_score = features["exact_interface_score"]

            result["query_type"] = query_type

            result["query_entities"] = entities

            # Backward compatibility
            result["query_entity"] = entities[0] if entities else None

            result["entity_score"] = entity_score

            result["entity_coverage"] = coverage

            result["intent_score"] = intent_score

            result["concept_score"] = concept_score

            result["section_score"] = section_score

            result["canonical_concept_score"] = canonical_concept_score

            result["section_type"] = features.get("section_type", "")

            result["interface_score"] = interface_score

            result["exact_interface_score"] = exact_interface_score

            # ==================================================
            # FINAL SCORE
            # ==================================================

            result["final_retrieval_score"] = (
                result["rrf_score"]
                + (ENTITY_WEIGHT * entity_score)
                + (COVERAGE_WEIGHT * coverage)
                + (INTENT_WEIGHT * intent_score)
                + (CONCEPT_WEIGHT * concept_score)
                + (SECTION_WEIGHT * section_score)
                + (CANONICAL_CONCEPT_WEIGHT * canonical_concept_score)
                + (INTERFACE_WEIGHT * interface_score)
                + (EXACT_INTERFACE_WEIGHT * exact_interface_score)
            )

        results.sort(key=lambda x: x["final_retrieval_score"], reverse=True)

        return results

    # ======================================================
    # PUBLIC SEARCH
    # ======================================================

    def search(self, query: str):

        # ==================================================
        # QUERY ANALYSIS
        # ==================================================

        query_analysis = self.query_analyzer.analyze(query)

        query_type = query_analysis["query_type"]

        # Use ALL entities.
        entities = query_analysis.get("entities", [])

        # ==================================================
        # FAISS
        # ==================================================

        vector_results = self.vector_search(query)

        # ==================================================
        # BM25
        # ==================================================

        bm25_results = self.bm25_search(query)

        # ==================================================
        # RRF
        # ==================================================

        hybrid_results = self.reciprocal_rank_fusion(vector_results, bm25_results)

        # ==================================================
        # QUERY-AWARE RANKING
        # ==================================================

        hybrid_results = self.apply_query_intent(
            hybrid_results, query, query_type, entities
        )

        # ==================================================
        # FINAL CANDIDATES
        # ==================================================

        return hybrid_results[: self.final_k]


# ==========================================================
# TEST OUTPUT
# ==========================================================


def print_results(query, results):

    print("\n")

    print("=" * 80)

    print(f"QUERY: {query}")

    print("=" * 80)

    if results:

        print(f"Query Type: " f"{results[0].get('query_type')}")

        print(f"Entities: " f"{results[0].get('query_entities')}")

    for rank, result in enumerate(results, start=1):

        chunk = result["chunk"]

        metadata = chunk["metadata"]

        print(
            f"\n[{rank}] "
            f"Final="
            f"{result['final_retrieval_score']:.6f} "
            f"RRF="
            f"{result['rrf_score']:.6f} "
            f"Entity="
            f"{result['entity_score']:.3f} "
            f"Coverage="
            f"{result['entity_coverage']:.2f} "
            f"Intent="
            f"{result['intent_score']:.3f} "
            f"Concept="
            f"{result['concept_score']:.2f} "
            f"Section="
            f"{result['section_score']:.2f} "
            f"CanonicalConcept="
            f"{result.get('canonical_concept_score', 0.0):.2f} "
            f"Interface="
            f"{result.get('interface_score', 0.0):.2f} "
            f"ExactInterface="
            f"{result.get('exact_interface_score', 0.0):.2f} "
            f"VectorRank="
            f"{result['vector_rank']} "
            f"BM25Rank="
            f"{result['bm25_rank']}"
        )

        print(f"Source: " f"{metadata.get('specification')}")

        section_number = metadata.get("section_number")

        section_title = metadata.get("section_title")

        if section_number:

            section_text = f"{section_number} " f"{section_title}"

        else:

            section_text = f"{section_title}"

        print(f"Section: " f"{section_text}")

        print("Text:")

        print(chunk["text"][:600])


# ==========================================================
# MAIN
# ==========================================================


def main():

    print("=" * 80)

    print(
        "HYBRID RETRIEVAL TEST — "
        "FAISS + BM25 + RRF + "
        "ENTITY + CONCEPT + INTENT + "
        "SECTION + INTERFACE"
    )

    print("=" * 80)

    retriever = HybridRetriever(vector_k=40, bm25_k=40, final_k=15, rrf_k=60)

    queries = [
        "What is the role of the AMF?",
        "What is the role of the SMF?",
        "What is the role of the UPF?",
        "What is the N4 interface?",
        "What is the N11 interface?",
        "What is PDU Session Establishment?",
        "How does PDU Session Establishment work?",
        "How does UE registration work in the 5G system?",
        "What is the difference between AMF and SMF?",
        "How does the AMF connect to the SMF?",
        "What is the capital of France?",
    ]

    for query in queries:

        results = retriever.search(query)

        print_results(query, results)


if __name__ == "__main__":

    main()
