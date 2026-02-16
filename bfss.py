graph={
    'A':['B','C'],
    'B':['D','E'],
    'C':['F'],
    'D':[],
}
visited=[]
def bfs(visited,graph_data,current_node):
    queue=[current_node]
    while queue:
        node=queue.pop(0)
        if node not in visited:
            print(node,end=" ")
            visited.append(node)
            queue.extend(graph_data.get(node,[]))

bfs(visited,graph,'B')