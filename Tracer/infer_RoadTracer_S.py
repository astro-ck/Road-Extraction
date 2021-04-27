import geom, graph
import model
import model_utils
import tileloader

import numpy

import os.path

import tensorflow as tf
import time

from remove_corners import remove_corners, read_graph

MAX_PATH_LENGTH = 500000
SEGMENT_LENGTH = 20
PATHS_PER_TILE_AXIS = 1
TILE_MODE = 'sat'
EXISTING_GRAPH_FNAME = None
DETECT_MODE = 'normal'
THRESHOLD_BRANCH = 0.4
THRESHOLD_FOLLOW = 0.4
WINDOW_SIZE = 256
SAVE_EXAMPLES = False
FOLLOW_TARGETS = False
TILE_SIZE = 1024


def vector_to_action(angle_outputs, stop_outputs, threshold):
    x = numpy.zeros((64,), dtype='float32')
    if stop_outputs[0] > threshold:
        x[numpy.argmax(angle_outputs)] = stop_outputs[0]
    return x4


def action_to_vector(v):
    angle_outputs = numpy.zeros((64,), dtype='float32')
    stop_outputs = numpy.zeros((2,), dtype='float32')
    count = 0
    for i in range(len(v)):
        if v[i] > 0.9:
            count += 1
    if count == 0:
        stop_outputs[1] = 1
    else:
        stop_outputs[0] = 1
        for i in range(len(v)):
            if v[i] > 0.9:
                angle_outputs[i] = 1.0 / count
    return angle_outputs, stop_outputs


def fix_outputs(batch_angle_outputs, batch_stop_outputs):
    if batch_angle_outputs.shape[1] == 64:
        return batch_angle_outputs, batch_stop_outputs
    elif batch_angle_outputs.shape[1] == 65:
        fixed_stop_outputs = numpy.zeros((batch_angle_outputs.shape[0], 2), dtype='float32')
        for i in range(batch_angle_outputs.shape[0]):
            if numpy.argmax(batch_angle_outputs[i, :]) == 64:
                fixed_stop_outputs[i, 1] = 1
            else:
                fixed_stop_outputs[i, 0] = 1
        return batch_angle_outputs[:, 0:64], fixed_stop_outputs
    else:
        raise Exception("bad angle_outputs length={}".format(len(angle_outputs)))


def score_accuracy(stop_targets, angle_targets, stop_outputs, angle_outputs, threshold, action_only=False):
    target_action = stop_targets[0] > threshold
    output_action = stop_outputs[0] > threshold
    if target_action != output_action:
        accuracy = 0.0
    elif not target_action:
        accuracy = 1.0
    elif action_only:
        accuracy = 1.0
    else:
        target_angle = numpy.argmax(angle_targets)
        output_angle = numpy.argmax(angle_outputs)
        angle_distance = abs(target_angle - output_angle)
        if angle_distance > 32:
            angle_distance = 64 - angle_distance
        if angle_distance > 16:
            accuracy = 0.0
        else:
            accuracy = 1.0 - float(angle_distance) / 16
    return accuracy


def eval(paths, m, session, starting_points_list, max_path_length=MAX_PATH_LENGTH, segment_length=SEGMENT_LENGTH, save=False,
         follow_targets=False, compute_targets=True, max_batch_size=model.BATCH_SIZE, window_size=WINDOW_SIZE,
         verbose=True, threshold_override=False):
    angle_losses = []
    detect_losses = []
    stop_losses = []
    losses = []
    accuracies = []
    path_lengths = {path_idx: 0 for path_idx in range(len(paths))}

    last_time = None
    big_time = None

    for len_it in range(99999999):
        if len_it % 1000 == 0 and verbose:
            print('it {}'.format(len_it))
            big_time = time.time()
        path_indices = []
        extension_vertices = []
        # get next road vertex
        for path_idx in range(len(paths)):
            # if total number of paths is greater than the maximum, then pass
            if path_lengths[path_idx] >= max_path_length:
                continue
            extension_vertex = paths[path_idx].pop()

            # if the next extension_vertex is none, then stop
            # in the final iter, extension vertex is none, them path_indices is none,
            # the program will stop
            if extension_vertex is None:
                continue
            path_indices.append(path_idx)
            # sum the path_length
            path_lengths[path_idx] += 1
            extension_vertices.append(extension_vertex)

            if len(path_indices) >= max_batch_size:
                break

        if len(path_indices) == 0:
            break

        batch_inputs = []
        batch_detect_targets = []
        batch_angle_targets = numpy.zeros((len(path_indices), 64), 'float32')
        batch_stop_targets = numpy.zeros((len(path_indices), 2), 'float32')

        for i in range(len(path_indices)):
            path_idx = path_indices[i]

            # 256*256*5  64*64*1
            path_input, path_detect_target = model_utils.make_path_input(paths[path_idx], extension_vertices[i],
                                                                         segment_length, window_size=window_size)

            batch_inputs.append(path_input)
            batch_detect_targets.append(path_detect_target)

            if compute_targets:
                targets = model_utils.compute_targets_by_best(paths[path_idx], extension_vertices[i], segment_length)
                angle_targets, stop_targets = action_to_vector(targets)
                batch_angle_targets[i, :] = angle_targets
                batch_stop_targets[i, :] = stop_targets

        feed_dict = {
            m.is_training: False,
            m.inputs: batch_inputs,
            m.angle_targets: batch_angle_targets,
            m.action_targets: batch_stop_targets,
            m.detect_targets: batch_detect_targets,
        }
        batch_angle_outputs, batch_stop_outputs, batch_detect_outputs, angle_loss, detect_loss, stop_loss, loss = session.run(
            [m.angle_outputs, m.action_outputs, m.detect_outputs, m.angle_loss, m.detect_loss, m.action_loss, m.loss],
            feed_dict=feed_dict)
        angle_losses.append(angle_loss)
        detect_losses.append(detect_loss)
        stop_losses.append(stop_loss)
        losses.append(loss)
        batch_angle_outputs, batch_stop_outputs = fix_outputs(batch_angle_outputs, batch_stop_outputs)

        # whether save result
        if save and len_it % 1 == 0:
            fname = '/data/temp/{}_'.format(len_it)
            save_angle_targets = batch_angle_targets[0, :]
            if not compute_targets:
                save_angle_targets = None
            model_utils.make_path_input(paths[path_indices[0]], extension_vertices[0], segment_length, fname=fname,
                                        angle_targets=save_angle_targets, angle_outputs=batch_angle_outputs[0, :],
                                        detect_output=batch_detect_outputs[0, :, :, 0], window_size=window_size)

        for i in range(len(path_indices)):
            path_idx = path_indices[i]
            if len(extension_vertices[i].out_edges) >= 2:
                threshold = THRESHOLD_BRANCH
                mode = 'branch'
            else:
                threshold = THRESHOLD_FOLLOW
                mode = 'follow'
            if threshold_override:
                threshold = threshold_override

            if follow_targets == True:
                x = vector_to_action(batch_angle_targets[i, :], batch_stop_targets[i, :], threshold=threshold)
            elif follow_targets == 'partial':
                # (a) always use stop_targets instead of stop_outputs
                # (b) if we are far away from graph, use angle_targets, otherwise use angle_outputs
                extension_vertex = batch_extension_vertices[i]
                if extension_vertex.edge_pos is None or extension_vertex.edge_pos.point().distance(
                        extension_vertex.point) > SEGMENT_LENGTH * 2:
                    x = vector_to_action(batch_angle_targets[i, :], batch_stop_targets[i, :], threshold=threshold)
                else:
                    x = vector_to_action(batch_angle_outputs[i, :], batch_stop_targets[i, :], threshold=threshold)
            elif follow_targets == 'npartial':
                # always move if gt says to move
                if batch_stop_outputs[i, 0] > threshold:
                    x = vector_to_action(batch_angle_outputs[i, :], batch_stop_outputs[i, :], threshold=threshold)
                else:
                    x = vector_to_action(batch_angle_outputs[i, :], batch_stop_targets[i, :], threshold=threshold)
            elif follow_targets == False:
                # 64, all other positions are 0 only the walk direction is p
                x = vector_to_action(batch_angle_outputs[i, :], batch_stop_outputs[i, :], threshold=threshold)
            else:
                raise Exception('invalid FOLLOW_TARGETS setting {}'.format(follow_targets))

            paths[path_idx].push(extension_vertices[i], x, segment_length, training=False, branch_threshold=0.01,
                                 follow_threshold=0.01)

            # score accuracy
            accuracy = score_accuracy(batch_stop_targets[i, :], batch_angle_targets[i, :], batch_stop_outputs[i, :],
                                      batch_angle_outputs[i, :], threshold)
            accuracies.append(accuracy)

    if save:
        paths[0].graph.save('out.graph')

    return numpy.mean(angle_losses), numpy.mean(detect_losses), numpy.mean(stop_losses), numpy.mean(
        losses), len_it, numpy.mean(accuracies)


def graph_filter(g, threshold=0.3, min_len=None):
    road_segments, _ = graph.get_graph_road_segments(g)
    bad_edges = set()
    for rs in road_segments:
        if min_len is not None and len(rs.edges) < min_len:
            bad_edges.update(rs.edges)
            continue
        probs = []
        if len(rs.edges) < 5 or True:
            for edge in rs.edges:
                if hasattr(edge, 'prob'):
                    probs.append(edge.prob)
        else:
            for edge in rs.edges[2:-2]:
                if hasattr(edge, 'prob'):
                    probs.append(edge.prob)
        if not probs:
            continue
        avg_prob = numpy.mean(probs)
        if avg_prob < threshold:
            bad_edges.update(rs.edges)
    print('filtering {} edges'.format(len(bad_edges)))
    ng = graph.Graph()
    vertex_map = {}
    for vertex in g.vertices:
        vertex_map[vertex] = ng.add_vertex(vertex.point)
    for edge in g.edges:
        if edge not in bad_edges:
            ng.add_edge(vertex_map[edge.src], vertex_map[edge.dst])
    return ng


if __name__ == '__main__':
    model_path = "../model/model_latest/model"
    test_sat_dir = "/data/test/sat/"
    BRANCH_THRESHOLD = 0.4
    FOLLOW_THRESHOLD = 0.4

    city_list = ['d']
    MANUAL_RELATIVE = geom.Point(-4, -4).scale(TILE_SIZE)
    for REGION in city_list:
        corner_file="/out/corner_detect/corners/"+REGION+"_boost8192.txt"
        output_fname = "/out/graph_infer/roadtracer-S/"
        if not os.path.isdir(output_fname):
            os.mkdir(output_fname)
        print("test city:"+ REGION)
        TILE_START = geom.Point(-4, -4).scale(TILE_SIZE)
        TILE_END = TILE_START.add(geom.Point(8, 8).scale(TILE_SIZE))

        tileloader.tile_dir = test_sat_dir

        tileloader.pytiles_path = "/json/pytiles.json"
        tileloader.startlocs_path = "/json/starting_locations.json"

        print('reading tiles')

        tiles = tileloader.Tiles(PATHS_PER_TILE_AXIS, SEGMENT_LENGTH, 16, TILE_MODE)

        print('initializing model')
        model.BATCH_SIZE = 1
        m = model.Model(tiles.num_input_channels())
        session = tf.Session()
        m.saver.restore(session, model_path)

        node_list=[] # save all nodes already explored

        if EXISTING_GRAPH_FNAME is None:
            # read
            rect = geom.Rectangle(TILE_START, TILE_END)
            tile_data = tiles.get_tile_data(REGION, rect)

            # generate path list
            path_list=[]
            start_points=[]
            with open(corner_file,"r") as f:
                for line in f.readlines():
                    temp=line.strip().split(",")
                    start_points.append([int(temp[0]),int(temp[1])])

            graph_num=0
            # s_pt=start_points[4]
            s_pt=[3003, 2695]

            pos3_point=geom.Point(s_pt[0]-4096,s_pt[1]-4096).add(MANUAL_RELATIVE)
            pos3_pos=pos3_point
            pos4_point = geom.Point(s_pt[0]+5, s_pt[1]).add(MANUAL_RELATIVE)
            pos4_pos = pos4_point
            start_loc = [{
                'point': pos3_point,
                'edge_pos': pos3_pos,
            }, {
                'point': pos4_point,
                'edge_pos': pos4_pos,
            }]
            tile_data['starting_locations'] = start_loc
            print("start loc:{}".format(start_loc))
            tile_data['gc'] = None
            path1 = model_utils.Path(tile_data['gc'], tile_data, start_loc=start_loc)

            compute_targets = SAVE_EXAMPLES or FOLLOW_TARGETS
            result = eval([path1], m, session,start_points, save=SAVE_EXAMPLES, compute_targets=compute_targets,
                          follow_targets=FOLLOW_TARGETS)

            save_path = output_fname+"{}.out.graph".format(REGION)
            path1.graph.save(save_path)

        else:
            g = graph.read_graph(EXISTING_GRAPH_FNAME)
            r = g.bounds()
            tile_data = {
                'region': REGION,
                'rect': r.add_tol(WINDOW_SIZE / 2),
                'search_rect': r,
                'cache': cache,
                'starting_locations': [],
            }
            path = model_utils.Path(None, tile_data, g=g)
            for vertex in g.vertices:
                path.prepend_search_vertex(vertex)
