from ..agents.subconscious_agent import SubconsciousAgent
def generate_opinion(agent,subject):
    opinion = ""
    memories = SubconsciousAgent.get_opinion_impacting_memories(agent,subject)

    return opinion
