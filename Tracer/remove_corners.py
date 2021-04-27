from rtree import index

def read_graph(graph_path):
    graph_index=[-2,-2]
    print("offset graph {}".format(graph_index))
    size=1024
    offset_x=graph_index[0]*size
    offset_y=graph_index[1]*size
    node_list=[]
    with open(graph_path,"r") as f:
        for line in f.readlines():
            temp=line.strip().split(" ")
            if len(temp)==2:
                node_list.append([int(temp[0])-offset_x, int(temp[1])-offset_y])
            else:
                print("a white line, jump out")
                break

    return node_list


def remove_corners(corner_list, node_list):
    class Corner_Task:
        def __init__(self, x, y, rect):
            self.x = x
            self.y = y
            self.rect = rect

    task_list = []
    for c in corner_list:
        # set the search box with radius of 60
        rect = (c[0] - 60, c[1] - 60, c[0] + 60, c[1] + 60)
        task = Corner_Task(c[0], c[1], rect)
        task_list.append(task)

    idx = index.Index()
    for i, node in enumerate(node_list):
        idx.insert(i, (node[0], node[1]))

    start_pts = []
    for t in task_list:
        cns = list(idx.intersection(t.rect, objects=True))
        if len(cns) < 1:
            start_pts.append([t.x, t.y])

    print("there are still {} start points left".format(len(start_pts)))
    return start_pts


if __name__=="__main__":    
    node_list=read_graph("out.graph")

    corner_list=[]
    with open("./points/chicago_gray.txt","r") as f:
        for line in f.readlines():
            temp=line.strip().split(",")
            corner_list.append([int(temp[0]),int(temp[1])])

    remove_corners(corner_list,node_list)
