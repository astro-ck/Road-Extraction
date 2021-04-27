import sys

sys.path.append("./discoverlib")
import os
from discoverlib import geom, graph
from rtree import index
import math

def generate_search_box(bounds,padding):
    if bounds.start.x + padding < bounds.end.x - padding:
        min_x = bounds.start.x + padding
        max_x = bounds.end.x - padding
    elif bounds.start.x + padding > bounds.end.x - padding:
        min_x = bounds.end.x - padding
        max_x = bounds.start.x + padding
    if bounds.start.y + padding < bounds.end.y - padding:
        min_y = bounds.start.y + padding
        max_y = bounds.end.y - padding
    elif bounds.start.y + padding > bounds.end.y - padding:
        min_y = bounds.end.y - padding
        max_y = bounds.start.y + padding
    return min_x,min_y,max_x,max_y


graph_dir="/out/graph_infer/c2/"
file_name = os.listdir(graph_dir)

# choose the largest as base graph
large_size = os.path.getsize(graph_dir + file_name[0])
large_id = 0
for i in range(1, len(file_name)):
    if os.path.getsize(graph_dir + file_name[i]) > large_size:
        large_size = os.path.getsize(graph_dir + file_name[i])
        large_id = i
print("the largest graph is {}".format(file_name[large_id]))
graph1 = graph.read_graph(graph_dir + file_name[large_id])
print("base on {}".format(file_name[large_id]))

# # choose the first one as base graph
# graph1 = graph.read_graph(graph_dir + file_name[0])
# print("base on {}".format(file_name[0]))
ind1 = index.Index()
id1 = 0
for edge1 in graph1.edges:
    bounds1 = edge1.bounds()
    # print(edge1.id)
    ind1.insert(edge1.id, (bounds1.start.x, bounds1.start.y, bounds1.end.x, bounds1.end.y))
    id1 += 1
print("total ids:{}".format(id1))

# reccord th
dislinked_edge = 0
for i in range(0, len(file_name)):
    if i != large_id: # except for base graph
        # graphs to be merged
        print(file_name[i])
        print("id={}".format(id1))
        graph2 = graph.read_graph(graph_dir + file_name[i])
        for edge in graph2.edges:
            if edge.id % 2 != 0:
                continue
            bounds = edge.bounds()
            padding = 0.01

            min_x, min_y, max_x, max_y=generate_search_box(bounds, padding)
            edge_ids = list(ind1.intersection((min_x, min_y, max_x, max_y), objects=True))
            # edge_ids=list(edge_ids)
            if len(edge_ids) < 1:
                # if dislinked_edge == 0:
                #     # link the disconnected area
                #     edges_id = list(ind1.index.intersection(
                #         (bounds.start.x - 40, bounds.start.y - 40, bounds.end.x + 40, bounds.end.y + 40)))
                #     if len(edges_id) > 0:
                #         graph1.add_vertex(edges_id[0].dst.point)
                #         pt_id = len(graph1.vertices)
                #         graph1.add_edge(graph1.vertices[pt_id - 2], graph1.vertices[pt_id - 1])
                #     dislinked_edge += 1
                # else:
                graph1.add_vertex(edge.src.point)
                graph1.add_vertex(edge.dst.point)
                pt_id = len(graph1.vertices)
                graph1.add_edge(graph1.vertices[pt_id - 2], graph1.vertices[pt_id - 1])
                new_edge = graph.Edge(len(graph1.edges), graph1.vertices[pt_id - 2], graph1.vertices[pt_id - 1])
                # print("lalala:{}".format(len(graph1.edges)))
                new_bounds = new_edge.bounds()
                ind1.insert(id1, (new_bounds.start.x, new_bounds.start.y, new_bounds.end.x, new_bounds.end.y))
                id1 += 1
            else:
                # min_x, min_y, max_x, max_y = generate_search_box(bounds, padding=0.001)
                # edge_ids = list(ind1.intersection((min_x, min_y, max_x, max_y), objects=True))
                if len(edge_ids) >= 1:
                    # print("edge bounds {} {} {} {}".format(bounds.start.x,bounds.start.y,bounds.end.x,bounds.end.y))
                    # for t_id in edge_ids:
                    #     print("id {}, box {}".format(t_id.id,t_id.bbox))
                    target_edge_sin = (edge.src.point.y-edge.dst.point.y)/math.sqrt((edge.src.point.y-edge.dst.point.y)**2+(edge.src.point.x-edge.dst.point.x)**2)
                    target_edge_angle = math.asin(target_edge_sin)

                    # print("previous edge id is:{}".format(previous_edge_id))
                    for base_id in edge_ids:
                        base_edge = graph1.edges[base_id.id]
                        base_edge_sin = (base_edge.src.point.y-base_edge.dst.point.y)/math.sqrt((base_edge.src.point.y-base_edge.dst.point.y)**2+(base_edge.src.point.x-base_edge.dst.point.x)**2)
                        base_edge_angle = math.asin(base_edge_sin)
                        angle_diff = abs(base_edge_angle - target_edge_angle)
                        if angle_diff > 0.6 and angle_diff < 3.15 / 2 - 0.6:
                            # print("add edge with angle diff {}".format(angle_diff))
                            graph1.add_vertex(edge.src.point)
                            graph1.add_vertex(edge.dst.point)
                            pt_id = len(graph1.vertices)
                            graph1.add_edge(graph1.vertices[pt_id - 2], graph1.vertices[pt_id - 1])
                            new_edge = graph.Edge(len(graph1.edges), graph1.vertices[pt_id - 2], graph1.vertices[pt_id - 1])
                            # print("lalala:{}".format(len(graph1.edges)))
                            new_bounds = new_edge.bounds()
                            ind1.insert(id1, (new_bounds.start.x, new_bounds.start.y, new_bounds.end.x, new_bounds.end.y))
                            id1 += 1
                            break

print("vertices length={}".format(len(graph1.vertices)))
print("edge length={}".format(len(graph1.edges)))
# graph1.edges[1].src.point.x
graph1.save("/out/graph_infer/line_merge_final/c2.final.graph")
print("done")
