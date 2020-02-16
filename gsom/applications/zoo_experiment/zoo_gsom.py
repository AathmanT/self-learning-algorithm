import time
import sys
import os
from os.path import join
from datetime import datetime
sys.path.append('../../')
# import cProfile

import data_parser as Parser
from util import utilities as Utils
from util import display as Display_Utils

from params import params as Params
from core4 import core_controller as Core


# GSOM config
SF = 0.83
# SF = 0.50

forget_threshold = 80  # To include forgetting, threshold should be < learning iterations.
temporal_contexts = 1  # If stationary data - keep this at 1
learning_itr = 100
smoothing_irt = 50
plot_for_itr = 4  # Unused parameter - just for visualization. Keep this as it is.

# File Config
dataset = 'zoo'
# data_filename = "data/creditcard-very-short.txt".replace('\\', '/')
data_filename = "data/zoo-mini.txt".replace('\\', '/')
experiment_id = 'Exp-new-gsom-' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
output_save_location = join('output/', experiment_id)


def generate_output_config(dataset, SF, forget_threshold):

    # Output data config
    output_save_filename = '{}_data_'.format(dataset)
    filename = output_save_filename + str(SF) + '_T_' + str(temporal_contexts) + '_mage_' + str(
        forget_threshold) + 'itr'
    plot_output_name = join(output_save_location, filename)

    # Generate output plot location
    output_loc = plot_output_name
    output_loc_images = join(output_loc, 'images/')
    if not os.path.exists(output_loc):
        os.makedirs(output_loc)
    if not os.path.exists(output_loc_images):
        os.makedirs(output_loc_images)

    return output_loc, output_loc_images


if __name__ == '__main__':
# if __name__ == 'zoo_gsom':
        # Init GSOM Parameters
        gsom_params = Params.GSOMParameters(SF, learning_itr, smoothing_irt, distance=Params.DistanceFunction.EUCLIDEAN,
                                            temporal_context_count=temporal_contexts, forget_itr_count=forget_threshold)
        generalise_params = Params.GeneraliseParameters(gsom_params)

        # Process the input files
        input_vector_database, labels, classes = Parser.InputParser.parse_input_zoo_data(data_filename, None)
        output_loc, output_loc_images = generate_output_config(dataset, SF, forget_threshold)

        # Setup the age threshold based on the input vector length
        generalise_params.setup_age_threshold(input_vector_database[0].shape[0])

        # Process the clustering algorithm algorithm
        controller = Core.Controller(generalise_params)
        controller_start = time.time()
        result_dict = controller.run(input_vector_database, plot_for_itr, classes, output_loc_images)
        # print(result_dict)
        # result_dict = cProfile.run('controller.run(input_vector_database, plot_for_itr, classes, output_loc_images)')
        print('Algorithms completed in', round(time.time() - controller_start, 2), '(s)')
        saved_name = Utils.Utilities.save_object(result_dict, join(output_loc, 'gsom_nodemap_SF-{}'.format(SF)))

        gsom_nodemap = result_dict[0]['gsom']

        from collections import Counter

        def vote(neighbors):
            class_counter = Counter()
            for neighbor in neighbors:
                class_counter[neighbor[2]] += 1
            return class_counter.most_common(1)[0][0]


        for key,value in gsom_nodemap.items():
            print(key," => ",[str(classes[lbl_id]) for lbl_id in value.get_mapped_labels()])
        # Display
        display = Display_Utils.Display(result_dict[0]['gsom'], None)
        display.setup_labels_for_gsom_nodemap(classes, 2, 'Latent Space of {} : SF={}'.format(dataset, SF),
                                              join(output_loc, 'latent_space_' + str(SF) + '_hitvalues'))
        display.setup_labels_for_gsom_nodemap(labels, 2, 'Latent Space of {} : SF={}'.format(dataset, SF),
                                              join(output_loc, 'latent_space_' + str(SF) + '_labels'))

        print('Completed.')
