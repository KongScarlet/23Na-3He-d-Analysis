import pfunk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import corner

fresco_path = './11_452_state_new_mix.in'

fresco_names = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6',
                'p1', 'p2', 'p3', 'p4', ('p5', 'p5'), ('p6', 'p6'), 'p4']

fresco_positions = [87, 88, 89, 95, 96, 97,
                    117, 118, 119, 120, (121, 129), (122, 130), 128]

model_inputs = [fresco_path, fresco_names, fresco_positions]
elastic_data_path = '../23Na_Normalized_to_Si_for_fresco.csv'
transfer_data_path = './11_452_first_min_CS.dat'

model = pfunk.model.Model(*model_inputs)
model.create_norm_prior([-10.0], [10.0])

init_values = np.array([117.31, 1.25, .65, 19.87, 1.25, .65,
                        88.1, 1.17, .74, 0.3, 1.32, .73, 12.3])
percents = np.array([.2, .2, .2, .2, .2, .2,
                     .1, .1, .1, .1, .1, .1, .1])

model.create_pot_prior(init_values, init_values*percents)
model.create_scatter_prior(widths=[0.10])
model.create_scatter_prior()
model.create_spec_prior([0.0], [1.5]) # C2S, position 1
model.create_spec_prior([1.0], [0.15], gaus=True) # D0, position 2
model.create_spec_prior(1,1,mixing_percent=True) # The mixing parameter, alpha, position 3
model.create_prior()

model.create_elastic_likelihood('fort.201', elastic_data_path, norm_index=0, scatter_index=4)
                                       # Note that you need the fresco cross sections for l=0 and l=2 to be read in.
model.create_two_l_transfer_likelihood('fort.202', 'fort.203', transfer_data_path,
                                       [1, 2], 3, norm_index=0, scatter_index=5) # this is the mixed l specific likelihood
                                       # Note [1, 2] means C2S x D0, while the "3" argument tells the likelihood where the alpha parameter is
model.create_likelihood()
model.x0[0] = 5.0
model.x0[1] = 5.0
model.x0[2] = 1.0
model.x0[3] = 1.0
model.x0[4] = 1.0
model.x0[5] = 1.0
