import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib import cm


def spiral_xy(i, spiral_num):
    """
    Create the data for a spiral.

    Arguments:
        i runs from 0 to 96
        spiral_num is 1 or -1
    """
    φ = i/16 * math.pi
    r = 6.5 * ((104 - i)/104)
    x = (r * math.cos(φ) * spiral_num)/13 + 0.5
    y = (r * math.sin(φ) * spiral_num)/13 + 0.5
    return (x, y)

def generate_spiral_data():
    a, b = [], []
    for i in range(97):
        x, y = spiral_xy(i, 1)
        a.append([x, y, 1]) 
        x, y = spiral_xy(i, -1)
        b.append([x, y, -1]) 
    a = np.array(a)
    b = np.array(b)
    data = np.concatenate((a, b), axis=0)
    scatter_plot(data)
    return data

def scatter_plot(data):
    # fig = plt.figure(figsize=(6,6))
    x, y, z = data[:,0], data[:,1], data[:,2]
    plt.scatter(x, y, c=z, marker = 'o', cmap = cm.jet )
    plt.show()

def eval_model(hme, data):
    misclassified = 0
    for i, data_point in enumerate(data):
        feature_vec = data_point[:2]
        y = data_point[2]
        hme.eval_weights(feature_vec)
        y_pred_vec = hme.eval_mus_and_return_final_pred(feature_vec)
        # if i<3:
        #     print(data_point, y_pred_vec)
        y_pred = -1 if y_pred_vec[0] >= 0.6 else 0
        y_pred = 1 if y_pred_vec[1] >= 0.6 else 0
        if y_pred != y:
            misclassified += 1
    return misclassified

class HME:
    def __init__(self, height=10, branching_factor=2, learning_rate=0.4,
    feature_size=2, output_size=2):
        self.height = height
        self.branching_factor = branching_factor
        self.learning_rate = learning_rate
        self.feature_size = feature_size
        self.output_size = output_size

        self.gating_net_param = dict()
        for i in range(height):
            for j in range(1, (self.branching_factor**i) + 1):
                self.gating_net_param[(i, j)] = (np.random.uniform(low=-1, high=1, size=(branching_factor, feature_size)), {}, {})

        self.expert_net_param = dict()
        for j in range(1, (self.branching_factor**height) + 1):
            self.expert_net_param[j] = (np.random.uniform(low=-1, high=1, size=(output_size, feature_size)), {}, {})

        self.mu_vals = dict()
        for i in range(0, self.height+1):
            self.mu_vals[i] = dict()

        self.weights = dict() 
        for i in range(1, self.height+1):
            self.weights[i] = dict()

        self.cond_posterior_probs = dict()
        for i in range(1, self.height+1):
            self.cond_posterior_probs[i] = dict()

        self.posterior_probs = dict()
        for i in range(0, self.height+1):
            self.posterior_probs[i] = dict()

    def softmax_fnc(self, param_mat, x_vec):
        dot_product = np.matmul(param_mat, np.transpose(x_vec))
        dot_product = dot_product - dot_product.max()   # To avoid overflows
        exp_dot_prod = np.exp(dot_product)
        output_prob_vec = exp_dot_prod/np.sum(exp_dot_prod, axis=0)
        return output_prob_vec

    def eval_weights(self, x_vec):
        for i in range(self.height):
            for j in range(1, (self.branching_factor**i) + 1):
                children_weights = self.softmax_fnc(self.gating_net_param[(i, j)][0], x_vec)
                children_weights = np.flip(children_weights)
                for k, weight in enumerate(children_weights):
                    self.weights[i+1][self.branching_factor*j-k] = weight

    def eval_mus_and_return_final_pred(self, x_vec):
        # Expert Networks
        for i in range(1, (self.branching_factor**self.height) + 1):
            k = self.softmax_fnc(self.expert_net_param[i][0], x_vec)
            # Thresholding to avoid matrix inversion issues during parameter updates
            k[k < 0.0001] = 0.0001
            k[k > 0.9999] = 0.9999
            self.mu_vals[self.height][i] = k

        # Gating Networks
        for i in range(self.height - 1, -1, -1):
            for j in range(1, (self.branching_factor**i) + 1):
                accumulator = 0
                for k in range(self.branching_factor):
                    accumulator += self.weights[i+1][self.branching_factor*j-k]*\
                                    self.mu_vals[i+1][self.branching_factor*j-k]

                # Thresholding to avoid matrix inversion issues during parameter updates
                accumulator[accumulator < 0.0001] = 0.0001
                accumulator[accumulator > 0.9999] = 0.9999
                self.mu_vals[i][j] = accumulator
                
        return self.mu_vals[0][1]

    def eval_conditional_posterior_prob(self, pos, x_vec, y_vec):
        weights_pos = self.weights[pos[0]][pos[1]]
        weights_pos_i = self.weights[pos[0]]    # Across the level. Dict
        mu_pos = np.dot(self.mu_vals[pos[0]][pos[1]], y_vec)
        mu_pos_i = self.mu_vals[pos[0]]         # Across the level. Dict

        sum_prod_of_weight_mu_across_level = 0
        for j in weights_pos_i:     # j is node position in the level
            sum_prod_of_weight_mu_across_level += weights_pos_i[j] * np.dot(mu_pos_i[j], y_vec)

        self.cond_posterior_probs[pos[0]][pos[1]] = (weights_pos*mu_pos)/sum_prod_of_weight_mu_across_level

        return self.cond_posterior_probs[pos[0]][pos[1]]

    def posterior_prob_helper(self, pos, x_vec, y_vec):
        if (pos[0] in self.posterior_probs) and (pos[1] in self.posterior_probs[pos[0]]):
            return self.posterior_probs[pos[0]][pos[1]] # These are the ones stored when evaluated for siblings
        elif pos == (0, 1):     # reached root node 
            return 1    # posterior prob of root = 1
        else:
            conditional_posterior_prob = self.eval_conditional_posterior_prob(pos, x_vec, y_vec)
            parent_vertical_pos = pos[0] - 1
            parent_horizontal_pos = math.ceil(pos[1]/self.branching_factor)
            parent_pos = (parent_vertical_pos, parent_horizontal_pos)
            # recursion
            parent_posterior_prob = self.posterior_prob_helper(parent_pos, x_vec, y_vec)
            if parent_posterior_prob < 0.0001:
                parent_posterior_prob = 0.0001
            self.posterior_probs[parent_pos[0]][parent_pos[1]] = parent_posterior_prob

            return conditional_posterior_prob*parent_posterior_prob

    def eval_joint_posterior_probs(self, x_vec, y_vec):
        for i in range(1, (self.branching_factor**self.height) + 1):
            posterior_prob = self.posterior_prob_helper((self.height, i), x_vec, y_vec)
            if posterior_prob < 0.0001:
                posterior_prob = 0.0001
            self.posterior_probs[self.height][i] = posterior_prob

    def update_parameters(self, x_vec, y, action):
        x_vec = x_vec.reshape(1, -1)  # row matrix
        # For Gating Networks
        for i in range(0, self.height):     # Updating parameters across all levels
            for j in range(1, (self.branching_factor**i) + 1):    #Updating parameters in a level
                for k in range(self.branching_factor):  # Updating parameter vectors of a node associated-
                    # -with each target output
                    # y_k is the conditional_prob of Kth child node
                    if action == 'accumulate':
                        child_pos_in_next_level = (j*self.branching_factor) - (self.branching_factor - 1 - k)  
                        y_k = self.cond_posterior_probs[i+1][child_pos_in_next_level]
                        mu_k = self.weights[i+1][child_pos_in_next_level]
                        weight = self.posterior_probs[i][j]
                    
                        if k not in self.gating_net_param[(i,j)][1]:
                            self.gating_net_param[(i,j)][1][k] = weight*mu_k*(1-mu_k)*np.matmul(np.transpose(x_vec), x_vec)
                            self.gating_net_param[(i,j)][2][k] = weight*(y_k-mu_k)*np.transpose(x_vec)
                        else:
                            self.gating_net_param[(i,j)][1][k] += weight*mu_k*(1-mu_k)*np.matmul(np.transpose(x_vec), x_vec)
                            self.gating_net_param[(i,j)][2][k] += weight*(y_k-mu_k)*np.transpose(x_vec)
                    elif action == 'update':
                        try:
                            self.gating_net_param[(i,j)][0][k] += np.transpose(self.learning_rate*\
                                    np.matmul( np.linalg.inv(self.gating_net_param[(i,j)][1][k]),\
                                            self.gating_net_param[(i,j)][2][k])).flatten() 
                        except:
                            self.gating_net_param[(i,j)][1].pop(k)
                            self.gating_net_param[(i,j)][2].pop(k)
                        
                                        
        # For Expert Networks
        for i in range(1, (self.branching_factor**self.height) + 1):
            for k in range(len(y)):  # Updating parameter vectors of a node associated with each target output
                if action == 'accumulate':
                    y_k = y[k]
                    mu_k = self.mu_vals[self.height][i][k]
                    weight = self.posterior_probs[self.height][i]
                
                    if k not in self.expert_net_param[i][1]:
                        self.expert_net_param[i][1][k] = weight*mu_k*(1-mu_k)*np.matmul(np.transpose(x_vec), x_vec)
                        self.expert_net_param[i][2][k] = weight*(y_k-mu_k)*np.transpose(x_vec)
                    else:
                        self.expert_net_param[i][1][k] += weight*mu_k*(1-mu_k)*np.matmul(np.transpose(x_vec), x_vec)
                        self.expert_net_param[i][2][k] += weight*(y_k-mu_k)*np.transpose(x_vec)
                elif action == 'update':
                    try:
                        self.expert_net_param[i][0][k] += np.transpose(self.learning_rate*\
                                    np.matmul( np.linalg.inv(self.expert_net_param[i][1][k]),\
                                            self.expert_net_param[i][2][k])).flatten() 
                    except:
                        self.expert_net_param[i][1].pop(k)
                        self.expert_net_param[i][2].pop(k)
                    
        return

if __name__ == '__main__':
    data = generate_spiral_data()
    hme = HME(height=5)
    np.random.shuffle(data)

    feature_vec = None 
    y_vec = None
    epochs = 1
    for i in range(epochs):
        print("Epoch: "+str(i))
        for j, data_point in enumerate(data):
            feature_vec = data_point[:2]
            y = data_point[2]
            y_vec = np.array([1, 0]) if y==-1 else np.array([0, 1])

            # Expectation (estimation) step
            hme.eval_weights(feature_vec)
            hme.eval_mus_and_return_final_pred(feature_vec)
            hme.eval_joint_posterior_probs(feature_vec, y_vec)

            # accumulate parameter update specific entities
            hme.update_parameters(feature_vec, y_vec, action='accumulate')

            #  posterior_probs evaluated recursively using Dynamic Programming.
            #  In each EM iteration need to evaluate them newly and not use prev. step vals.
            #  Thus disposing them
            for k in range(0, hme.height+1):
                hme.posterior_probs[k] = dict()

        # maximization (update parameters), feature_vec, y_vec passed for dimensions
        hme.update_parameters(feature_vec, y_vec, action='update')

        misclassified = eval_model(hme, data)
        print("Misclassified {} out of {} in Training data".format(misclassified,len(data)))

    # The points in the test set are offset vertically from the points in the
    #  learning set by 0.1 as specified in the paper.
    test_data = data
    test_data[:,1] -= 0.1
    misclassified = eval_model(hme, test_data)
    print("Misclassified {} out of {} in Test data".format(misclassified,len(test_data)))




