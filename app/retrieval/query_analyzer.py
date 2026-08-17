import re


class QueryAnalyzer:
    """
    Lightweight deterministic analyzer for telecom RAG queries.

    Responsibilities:
        1. Normalize the query.
        2. Detect query intent/type.
        3. Extract telecom entities.
        4. Detect domain concepts.

    No LLM is used here. The analyzer must remain deterministic
    and inexpensive because it runs before retrieval.
    """

    # ==========================================================
    # KNOWN TELECOM ENTITIES
    # ==========================================================

    KNOWN_ENTITIES = [
        # ------------------------------------------------------
        # 5G Core Network Functions
        # ------------------------------------------------------
        "AMF",
        "SMF",
        "UPF",
        "AUSF",
        "UDM",
        "UDR",
        "PCF",
        "NRF",
        "NSSF",
        "NEF",
        "AF",
        "CHF",
        "NWDAF",
        "SCP",
        # ------------------------------------------------------
        # Interfaces / Reference Points
        # ------------------------------------------------------
        "N1",
        "N2",
        "N3",
        "N4",
        "N6",
        "N7",
        "N8",
        "N9",
        "N10",
        "N11",
        "N12",
        "N13",
        "N14",
        "N15",
        "N16",
        # ------------------------------------------------------
        # Other common telecom entities
        # ------------------------------------------------------
        "UE",
        "NG-RAN",
        "RAN",
        "5GC",
        "5G-AN",
        "DN",
        "gNB",
        "S-NSSAI",
        "DNN",
    ]

    # ==========================================================
    # QUERY TYPE PATTERNS
    # ==========================================================

    QUERY_PATTERNS = {
        # Check before DEFINITION because:
        # "What is the difference between AMF and SMF?"
        # contains "what is".
        "COMPARISON": [
            r"\bdifference between\b",
            r"\bdifferent between\b",
            r"\bcompare\b",
            r"\bcomparison between\b",
            r"\bhow .* differ\b",
            r"\bvs\.?\b",
            r"\bversus\b",
            r"\bwhat .* difference\b",
        ],
        "ROLE": [
            r"\brole of\b",
            r"\bfunction of\b",
            r"\bfunctions of\b",
            r"\bresponsibility of\b",
            r"\bresponsibilities of\b",
            r"\bwhat does .* do\b",
            r"\bwhat is .* responsible for\b",
        ],
        "INTERFACE": [
            r"\binterface\b",
            r"\breference point\b",
            r"\bwhat does .* interface\b",
            r"\bhow does .* connect\b",
            r"\bconnection between\b",
        ],
        "PROCEDURE": [
            r"\bhow does .* work\b",
            r"\bhow .* works\b",
            r"\bhow to\b",
            r"\bsteps\b",
            r"\bprocedure\b",
            r"\bprocess\b",
            r"\bflow\b",
            r"\bestablishment\b",
            r"\brelease procedure\b",
            r"\bmodification procedure\b",
            r"\bregistration\b",
        ],
        "DEFINITION": [
            r"\bwhat is\b",
            r"\bwhat are\b",
            r"\bdefine\b",
            r"\bdefinition of\b",
            r"\bmeaning of\b",
            r"\bwhat does .* mean\b",
        ],
    }

    # ==========================================================
    # DOMAIN CONCEPT PATTERNS
    # ==========================================================

    # More specific concepts come first.
    DOMAIN_CONCEPTS = {
        "PDU_SESSION_ESTABLISHMENT": [
            "pdu session establishment procedure",
            "pdu session establishment",
            "establishment of a pdu session",
            "establish a pdu session",
        ],
        "REGISTRATION": [
            "initial registration",
            "5gs registration",
            "5g registration",
            "ue registration",
            "registration request",
            "registration procedure",
            "registration management",
            "registration",
        ],
        "SERVICE_REQUEST": [
            "service request procedure",
            "service request",
        ],
        "DEREGISTRATION": [
            "deregistration procedure",
            "de-registration procedure",
            "deregistration",
            "de-registration",
        ],
        "HANDOVER": [
            "xn handover",
            "n2 handover",
            "handover procedure",
            "handover",
        ],
        "PAGING": [
            "paging procedure",
            "paging",
        ],
        "AUTHENTICATION": [
            "primary authentication",
            "authentication procedure",
            "authentication",
        ],
    }

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(self):
        pass

    # ==========================================================
    # NORMALIZATION
    # ==========================================================

    @staticmethod
    def normalize_query(query: str) -> str:
        """Normalize whitespace while preserving the actual query."""

        if query is None:
            return ""

        query = str(query).strip()

        query = re.sub(r"\s+", " ", query)

        return query

    # ==========================================================
    # ENTITY EXTRACTION
    # ==========================================================

    def extract_entities(self, query: str):
        """
        Extract known telecom entities.

        Examples:
            What is the role of the AMF?
                -> ["AMF"]

            What is the difference between AMF and SMF?
                -> ["AMF", "SMF"]

            What is the N11 interface?
                -> ["N11"]
        """

        query_upper = query.upper()

        found = []

        # Longest first, e.g. NG-RAN before shorter terms.
        entities = sorted(self.KNOWN_ENTITIES, key=len, reverse=True)

        for entity in entities:

            pattern = r"(?<![A-Z0-9])" + re.escape(entity.upper()) + r"(?![A-Z0-9])"

            if re.search(pattern, query_upper):

                if entity.upper() not in {item.upper() for item in found}:
                    found.append(entity)

        return found

    # ==========================================================
    # PRIMARY ENTITY
    # ==========================================================

    @staticmethod
    def primary_entity(entities):

        if not entities:
            return None

        return entities[0]

    # ==========================================================
    # QUERY TYPE
    # ==========================================================

    def detect_query_type(self, query: str):

        query_lower = query.lower()

        priority = [
            "COMPARISON",
            "ROLE",
            "INTERFACE",
            "PROCEDURE",
            "DEFINITION",
        ]

        for query_type in priority:

            for pattern in self.QUERY_PATTERNS[query_type]:

                if re.search(pattern, query_lower):
                    return query_type

        return "GENERAL"

    # ==========================================================
    # DOMAIN CONCEPT
    # ==========================================================

    def detect_concept(self, query: str):

        query_lower = query.lower()

        # Check longest phrase first.
        all_candidates = []

        for concept, phrases in self.DOMAIN_CONCEPTS.items():
            for phrase in phrases:
                all_candidates.append((len(phrase), concept, phrase))

        all_candidates.sort(reverse=True)

        for _, concept, phrase in all_candidates:

            if phrase in query_lower:
                return concept

        return None

    # ==========================================================
    # ANALYZE
    # ==========================================================

    def analyze(self, query: str):

        query = self.normalize_query(query)

        query_type = self.detect_query_type(query)

        entities = self.extract_entities(query)

        concept = self.detect_concept(query)

        entity = self.primary_entity(entities)

        return {
            "query": query,
            "query_type": query_type,
            # Backward compatibility
            "entity": entity,
            # Multi-entity support
            "entities": entities,
            # Domain-aware retrieval
            "concept": concept,
            "entity_count": len(entities),
            "is_comparison": (query_type == "COMPARISON"),
            "is_telecom_query": (len(entities) > 0 or concept is not None),
        }


# ==========================================================
# TEST
# ==========================================================


def main():

    analyzer = QueryAnalyzer()

    queries = [
        "What is the role of the AMF?",
        "What is the role of the SMF?",
        "What is the role of the UPF?",
        "What is the N4 interface?",
        "What is the N11 interface?",
        "What is PDU Session Establishment?",
        "How does PDU Session Establishment work?",
        "How does UE registration work in the 5G system?",
        "What happens during initial registration?",
        "What is a registration request?",
        "What is the difference between AMF and SMF?",
        "Compare AMF vs SMF",
        "How does the AMF connect to the SMF?",
        "What is the capital of France?",
    ]

    print("=" * 80)
    print("QUERY ANALYZER TEST")
    print("=" * 80)

    for query in queries:

        result = analyzer.analyze(query)

        print()
        print(query)

        print(f"  Type:          " f"{result['query_type']}")

        print(f"  Entity:        " f"{result['entity']}")

        print(f"  Entities:      " f"{result['entities']}")

        print(f"  Concept:       " f"{result['concept']}")

        print(f"  Entity count:  " f"{result['entity_count']}")

        print(f"  Telecom query: " f"{result['is_telecom_query']}")


if __name__ == "__main__":
    main()
