graph={
    'A':['B','C'],
    'B':['D','E'],
    'C':['F'],
    'D':[],
}
visited=[]
def dfs(visited,graph_data,current_node):
    if current_node not in visited:
        print(current_node,end=" ")
        visited.append(current_node)
        for nei in graph_data.get(current_node,[]):
            dfs(visited,graph_data,nei)

dfs(visited,graph,'A')