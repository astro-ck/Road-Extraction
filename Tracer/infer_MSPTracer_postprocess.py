import geom, graph

graph_dir="/data/cities/out/graph_infer/roadtracer-M/"
graph_name_list = ["amsterdam", "chicago", "denver", "la", "montreal", "paris", "pittsburgh",
                   "saltlakecity", "san diego", "tokyo", "toronto", "vancouver"]
for graph_name in graph_name_list:
    print("current graph:"+graph_name)
    ori_graph=graph.read_graph(graph_dir+graph_name+".infer.graph")
    total_number=0
    total_connect=0
    new_edge_list=[]
    seg_pair_list=[]
    for i, edge in enumerate(ori_graph.edges):
        if edge.src.id==0:
            length=edge.segment().length()
            if length > 70:
                seg_pair_list.append(edge.dst)
                if len(seg_pair_list)==2:
                    edge1=graph.Edge(len(new_edge_list),seg_pair_list[0],seg_pair_list[1])
                    if edge1.segment().length()<61:
                        print("connect with length {}".format(edge1.segment().length()))
                        total_connect+=1
                        new_edge_list.append(edge1)
                        new_edge_list.append(graph.Edge(len(new_edge_list)+1,seg_pair_list[1],seg_pair_list[0]))
                    seg_pair_list=[]

                print("delete edge with length={}".format(length))
                total_number+=1
                continue
            else:
                new_edge_list.append(edge)
        if edge.dst.id==0:
            length=edge.segment().length()
            if length > 70:
                total_number+=1
                continue
            else:
                new_edge_list.append(edge)

        new_edge_list.append(edge)
    print("total delete number {}".format(total_number))
    print("total connect number {}".format(total_connect))
    ori_graph.edges=new_edge_list
    ori_graph.save(graph_dir+graph_name+".out.graph")
    print("before/after edge numbers={}/{}".format(i,len(new_edge_list)))
