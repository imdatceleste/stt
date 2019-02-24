# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

"""
Model frozen tool
Copyright (c) 2017 DTMS converting communication GmbH (dtms)

This tool is based on Blog article https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
Model restore method was changed. Operation list output was added. 

All rights reserved.
Created: 2017-10-16 12:00 CST, KA
"""

import os 
import argparse
import tensorflow as tf

def freeze_graph(model_dir, ckpt_name, output_node_names):
    """Extract the sub graph defined by the output nodes and convert all its variables into constant 
    Params:
        model_dir: the root folder containing the checkpoint state file
        ckpt_name: model name to restore
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError("Export directory doesn't exists. Please specify an export directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1
    
    # We precise the required file names
    model_name = os.path.join(model_dir, ckpt_name)
    graph_name = model_name + ".meta"

    output_graph = os.path.join(model_dir, "frozen_model.pb")

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(graph_name, clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, model_name)

        # We can verify that we can access the list of operations in the graph
        for op in sess.graph.get_operations():
            print(op.name)        

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", help="Model folder to export")
    parser.add_argument("--ckpt_name", type=str, default="", help="Model checkpoint name to restore")
    parser.add_argument("--output_node_names", type=str, default="", help="The name of the output nodes, comma separated.")
    args = parser.parse_args()

    freeze_graph(args.model_dir, args.ckpt_name, args.output_node_names)