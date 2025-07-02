for state in graph.get_state_histor(config):
    print("NUm Messages",len(state.values["messages"]),"Next",state.next)
    if len(state.values["message"])===6:
        