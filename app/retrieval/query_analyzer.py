from pathlib import Path
import re


class QueryAnalyzer:

    # ==========================================================
    # KNOWN TELECOM ENTITIES
    # ==========================================================

    KNOWN_ENTITIES = [

        # Network Functions
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

        # Interfaces / Reference Points
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

        # Other common telecom entities
        "UE",
        "RAN",
        "NG-RAN",
        "5GC",
        "5G-AN",
        "DN",
        "UPF",
        "gNB",
        "S-NSSAI",
        "DNN",
      
    ]


    # ==========================================================
    # QUERY TYPE PATTERNS
    # ==========================================================

    QUERY_PATTERNS = {

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
    # INITIALIZATION
    # ==========================================================

    def __init__(self):

        pass


    # ==========================================================
    # NORMALIZE QUERY
    # ==========================================================

    @staticmethod
    def normalize_query(
        query: str
    ):

        query = query.strip()

        query = re.sub(
            r"\s+",
            " ",
            query
        )

        return query


    # ==========================================================
    # EXTRACT ENTITIES
    # ==========================================================

    def extract_entities(
        self,
        query: str
    ):

        """
        Extract all known telecom entities.

        Example:

            What is the role of the AMF?
                -> ["AMF"]

            What is the difference between AMF and SMF?
                -> ["AMF", "SMF"]

            What is the N11 interface?
                -> ["N11"]

        """

        query_upper = query.upper()

        found = []


        # ------------------------------------------------------
        # Sort longest first.
        #
        # This prevents things like:
        #
        # NG-RAN
        #
        # being incorrectly interpreted before the complete
        # entity is checked.
        # ------------------------------------------------------

        entities = sorted(
            self.KNOWN_ENTITIES,
            key=len,
            reverse=True
        )


        for entity in entities:

            pattern = (
                r"(?<![A-Z0-9])"
                + re.escape(
                    entity.upper()
                )
                + r"(?![A-Z0-9])"
            )


            if re.search(
                pattern,
                query_upper
            ):

                if entity.upper() not in [
                    x.upper()
                    for x in found
                ]:

                    found.append(
                        entity
                    )


        return found


    # ==========================================================
    # EXTRACT PRIMARY ENTITY
    # ==========================================================

    @staticmethod
    def primary_entity(
        entities
    ):

        if not entities:

            return None

        return entities[0]


    # ==========================================================
    # DETECT QUERY TYPE
    # ==========================================================

    def detect_query_type(
        self,
        query: str
    ):

        query_lower = query.lower()


        # ------------------------------------------------------
        # Order matters.
        #
        # Comparison must be checked before definition because:
        #
        # "What is the difference between AMF and SMF?"
        #
        # contains "what is".
        # ------------------------------------------------------

        priority = [

            "COMPARISON",

            "ROLE",

            "INTERFACE",

            "PROCEDURE",

            "DEFINITION",
        ]


        for query_type in priority:

            patterns = self.QUERY_PATTERNS[
                query_type
            ]


            for pattern in patterns:

                if re.search(
                    pattern,
                    query_lower
                ):

                    return query_type


        return "GENERAL"


    # ==========================================================
    # ANALYZE
    # ==========================================================

    def analyze(
        self,
        query: str
    ):

        query = self.normalize_query(
            query
        )


        query_type = (
            self.detect_query_type(
                query
            )
        )


        entities = (
            self.extract_entities(
                query
            )
        )


        entity = (
            self.primary_entity(
                entities
            )
        )


        return {

            "query":
                query,

            "query_type":
                query_type,

            # Backward compatibility
            "entity":
                entity,

            # New multi-entity field
            "entities":
                entities,

            "entity_count":
                len(entities),

            "is_comparison":
                query_type == "COMPARISON",
        }


# ==========================================================
# TEST
# ==========================================================

def main():

    analyzer = QueryAnalyzer()


    queries = [

        "What is the role of the AMF?",

        "What is the N11 interface?",

        "What is PDU Session Establishment?",

        "How does PDU Session Establishment work?",

        "What is the role of the SMF?",

        "What is the difference between AMF and SMF?",

        "Compare AMF vs SMF",

        "How does the AMF connect to the SMF?",
    ]


    print("=" * 80)

    print(
        "QUERY ANALYZER TEST"
    )

    print("=" * 80)


    for query in queries:

        result = analyzer.analyze(
            query
        )


        print()

        print(query)

        print(
            f"  Type: "
            f"{result['query_type']}"
        )

        print(
            f"  Entity: "
            f"{result['entity']}"
        )

        print(
            f"  Entities: "
            f"{result['entities']}"
        )

        print(
            f"  Count: "
            f"{result['entity_count']}"
        )


if __name__ == "__main__":

    main()