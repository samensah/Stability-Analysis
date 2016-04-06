"""
Copyright (C) 2014 Dallas Card

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
 to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.


Description:
Given unreliable observations of item classes by multiple observers,
determine the most likely true class for each item, class marginals,
and  individual error rates for each observer, using Expectation Maximization


References:
( Dawid and Skene (1979). Maximum Likelihood Estimation of Observer
Error-Rates Using the EM Algorithm. Journal of the Royal Statistical Society.
Series C (Applied Statistics), Vol. 28, No. 1, pp. 20-28. 
"""

import numpy as np
import sys
from kmeans_responses import responses
#from create_reliability_from_csv import responses

"""
Function: main()
    Run the EM estimator on the data from the Dawid-Skene paper
"""

text_file = open("em_output.txt", "w")
reliability_file = open("reliability.txt", "w")


def main():
    # load the data from the paper
    #responses = generate_sample_data()
    # run EM
    run(responses)

"""
Function: dawid_skene()
    Run the Dawid-Skene estimator on response data
Input:
    responses: a dictionary object of responses:
        {items: {users: [labels]}}
    tol: tolerance required for convergence of EM
    max_iter: maximum number of iterations of EM
"""
def run(responses, tol=0.00001, max_iter=100, init='average'):
    # convert responses to counts
    (items, users, classes, counts) = responses_to_counts(responses)
    text_file.write("Num of Items: {0}\n".format(len(items)))
    #print "num Items:", len(items)
    text_file.write("Num of Users: {0}\n".format(len(users)))
    #print "Users:", users
    text_file.write("Classes: {0}\n\n".format(classes))
    #print "Classes:", classes

    # initialize
    iter = 0
    converged = False
    old_class_marginals = None
    old_error_rates = None

    item_classes = initialize(counts)
    text_file.write("{0}".format("Iter\tlog-likelihood\tdelta-CM\tdelta-ER\n"))
    #print "Iter\tlog-likelihood\tdelta-CM\tdelta-ER"

    # while not converged do:
    while not converged:
        iter += 1

        # M-step
        (class_marginals, error_rates) = m_step(counts, item_classes)

        # E-setp
        item_classes = e_step(counts, class_marginals, error_rates)

        # check likelihood
        log_L = calc_likelihood(counts, class_marginals, error_rates)

        # check for convergence
        if old_class_marginals is not None:
            class_marginals_diff = np.sum(np.abs(class_marginals - old_class_marginals))
            error_rates_diff = np.sum(np.abs(error_rates - old_error_rates))
            #print iter ,'\t', log_L, '\t%.6f\t%.6f' % (class_marginals_diff, error_rates_diff)
            text_file.write("{0}\t{1}\t{2}\t{3}\n".format(iter, log_L, class_marginals_diff, error_rates_diff))
            if (class_marginals_diff < tol and error_rates_diff < tol) or iter > max_iter:
                converged = True
        else:
            #print iter ,'\t', log_L
            text_file.write("{0}\t{1}\n".format(iter, log_L))

        # update current values
        old_class_marginals = class_marginals
        old_error_rates = error_rates
    text_file.write("\n")
    # Print final results
    np.set_printoptions(precision=2, suppress=True)
    #print "Class marginals"
    text_file.write("Class marginals\n")
    #print class_marginals
    text_file.write("{0}\n".format(class_marginals))
    text_file.write("\n")
    #print "Error rates"
    text_file.write("Individual Error Rates\n")
    text_file.write("Diagonal is the probability of user choosing a True Label\n")
    #print error_rates
    for i in range(len(error_rates)):
        text_file.write("User {0}:\n{1}\n".format(i+1,np.array(error_rates[i])))

    text_file.write("\n")

    #print "Incidence-of-error rates"
    text_file.write("Incidence-of-error rates\n")
    text_file.write("Sum of diagonal elements is the probability of a correct allocation by user \n")
    [nItems, nUsers, nClasses] = np.shape(counts)
    for k in range(nUsers):
        #print class_marginals * error_rates[k,:,:]
        text_file.write("User {0}:\n{1}\n".format(k+1, class_marginals * error_rates[k,:,:] ))
        diag_sum = np.trace(class_marginals * error_rates[k,:,:])
        reliability_file.write("{0}\t{1:.4f}\n".format(k+1,diag_sum))

    np.set_printoptions(precision=4, suppress=True)
    text_file.write("\n")

    #print "Item classes"
    text_file.write("Item classes\n")
    text_file.write("Estimated probabilities for True labels for each Item \n")
    for i in range(nItems):
        #print items[i], item_classes[i,:]
        text_file.write("{0}{1}\n".format(items[i], item_classes[i,:]))

    text_file.close()
    #return (items, users, classes, counts, class_marginals, error_rates, item_classes)

"""
Function: responses_to_counts()
    Convert a matrix of annotations to count data
Inputs:
    responses: dictionary of responses {item:{users:[responses]}}
Return:
    items: list of items
    users: list of users
    classes: list of possible item classes
    counts: 3d array of counts: [items x users x classes]
"""
def responses_to_counts(responses):
    items = responses.keys()
    items.sort()
    nItems = len(items)

    # determine the users and classes
    users = set()
    classes = set()
    for i in items:
        i_users = responses[i].keys()
        for k in i_users:
            if k not in users:
                users.add(k)
            ik_responses = responses[i][k]
            classes.update(ik_responses)

    classes = list(classes)
    classes.sort()
    nClasses = len(classes)

    users = list(users)
    users.sort()
    nUsers = len(users)

    # create a 3d array to hold counts
    counts = np.zeros([nItems, nUsers, nClasses])

    # convert responses to counts
    for item in items:
        i = items.index(item)
        for user in responses[item].keys():
            k = users.index(user)
            for response in responses[item][user]:
                j = classes.index(response)
                counts[i,k,j] += 1


    return (items, users, classes, counts)


"""
Function: initialize()
    Get initial estimates for the true item classes using counts
    see equation 3.1 in Dawid-Skene (1979)
Input:
    counts: counts of the number of times each response was received
        by each user from each item: [items x users x classes]
Returns:
    item_classes: matrix of estimates of true item classes:
        [items x responses]
"""
def initialize(counts):
    [nItems, nUsers, nClasses] = np.shape(counts)
    # sum over users
    response_sums = np.sum(counts,1)
    # create an empty array
    item_classes = np.zeros([nItems, nClasses])
    # for each item, take the average number of observations in each class
    for p in range(nItems):
        item_classes[p,:] = response_sums[p,:] / np.sum(response_sums[p,:],dtype=float)
    #print item_classes
    return item_classes


"""
Function: m_step()
    Get estimates for the prior class probabilities (p_j) and the error
    rates (pi_jkl) using MLE with current estimates of true item classes
    See equations 2.3 and 2.4 in Dawid-Skene (1979)
Input:
    counts: Array of how many times each response was received
        by each user from each item
    item_classes: Matrix of current assignments of items to classes
Returns:
    p_j: class marginals [classes]
    pi_kjl: error rates - the probability of user k receiving
        response l from a item in class j [users, classes, classes]
"""
def m_step(counts, item_classes):
    [nItems, nUsers, nClasses] = np.shape(counts)

    # compute class marginals
    class_marginals = np.sum(item_classes,0)/float(nItems)

    # compute error rates
    error_rates = np.zeros([nUsers, nClasses, nClasses])
    for k in range(nUsers):
        for j in range(nClasses):
            for l in range(nClasses):
                error_rates[k, j, l] = np.dot(item_classes[:,j], counts[:,k,l])
            # normalize by summing over all observation classes
            sum_over_responses = np.sum(error_rates[k,j,:])
            if sum_over_responses > 0:
                error_rates[k,j,:] = error_rates[k,j,:]/float(sum_over_responses)

    return (class_marginals, error_rates)


"""
Function: e_step()
    Determine the probability of each item belonging to each class,
    given current ML estimates of the parameters from the M-step
    See equation 2.5 in Dawid-Skene (1979)
Inputs:
    counts: Array of how many times each response was received
        by each user from each item
    class_marginals: probability of a random item belonging to each class
    error_rates: probability of user k assigning a item in class j
        to class l [users, classes, classes]
Returns:
    item_classes: Soft assignments of items to classes
        [items x classes]
"""
def e_step(counts, class_marginals, error_rates):
    [nItems, nUsers, nClasses] = np.shape(counts)

    item_classes = np.zeros([nItems, nClasses])

    for i in range(nItems):
        for j in range(nClasses):
            estimate = class_marginals[j]
            estimate *= np.prod(np.power(error_rates[:,j,:], counts[i,:,:]))

            item_classes[i,j] = estimate
        # normalize error rates by dividing by the sum over all observation classes
        item_sum = np.sum(item_classes[i,:])
        if item_sum > 0:
            item_classes[i,:] = item_classes[i,:]/float(item_sum)

    return item_classes


"""
Function: calc_likelihood()
    Calculate the likelihood given the current parameter estimates
    This should go up monotonically as EM proceeds
    See equation 2.7 in Dawid-Skene (1979)
Inputs:
    counts: Array of how many times each response was received
        by each user from each item
    class_marginals: probability of a random item belonging to each class
    error_rates: probability of user k assigning a item in class j
        to class l [users, classes, classes]
Returns:
    Likelihood given current parameter estimates
"""
def calc_likelihood(counts, class_marginals, error_rates):
    [nItems, nUsers, nClasses] = np.shape(counts)
    log_L = 0.0

    for i in range(nItems):
        item_likelihood = 0.0
        for j in range(nClasses):

            class_prior = class_marginals[j]
            item_class_likelihood = np.prod(np.power(error_rates[:,j,:], counts[i,:,:]))
            item_class_posterior = class_prior * item_class_likelihood
            item_likelihood += item_class_posterior

        temp = log_L + np.log(item_likelihood)

        if np.isnan(temp) or np.isinf(temp):
            print i, log_L, np.log(item_likelihood), temp
            sys.exit()

        log_L = temp

    return log_L


"""
Function: generate_sample_data()
    Generate the data from Table 1 in Dawid-Skene (1979) in the proper format
"""
"""
def generate_sample_data():
    responses = {
                 1: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 2: {1:[3,3,3], 2:[4], 3:[3], 4:[3], 5:[4]},
                 3: {1:[1,1,2], 2:[2], 3:[1], 4:[2], 5:[2]},
                 4: {1:[2,2,2], 2:[3], 3:[1], 4:[2], 5:[1]},
                 5: {1:[2,2,2], 2:[3], 3:[2], 4:[2], 5:[2]},
                 6: {1:[2,2,2], 2:[3], 3:[3], 4:[2], 5:[2]},
                 7: {1:[1,2,2], 2:[2], 3:[1], 4:[1], 5:[1]},
                 8: {1:[3,3,3], 2:[3], 3:[4], 4:[3], 5:[3]},
                 9: {1:[2,2,2], 2:[2], 3:[2], 4:[2], 5:[3]},
                 10: {1:[2,3,2], 2:[2], 3:[2], 4:[2], 5:[3]},
                 11: {1:[4,4,4], 2:[4], 3:[4], 4:[4], 5:[4]},
                 12: {1:[2,2,2], 2:[3], 3:[3], 4:[4], 5:[3]},
                 13: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 14: {1:[2,2,2], 2:[3], 3:[2], 4:[1], 5:[2]},
                 15: {1:[1,2,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 16: {1:[1,1,1], 2:[2], 3:[1], 4:[1], 5:[1]},
                 17: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 18: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 19: {1:[2,2,2], 2:[2], 3:[2], 4:[2], 5:[1]},
                 20: {1:[2,2,2], 2:[1], 3:[3], 4:[2], 5:[2]},
                 21: {1:[2,2,2], 2:[2], 3:[2], 4:[2], 5:[2]},
                 22: {1:[2,2,2], 2:[2], 3:[2], 4:[2], 5:[1]},
                 23: {1:[2,2,2], 2:[3], 3:[2], 4:[2], 5:[2]},
                 24: {1:[2,2,1], 2:[2], 3:[2], 4:[2], 5:[2]},
                 25: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 26: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 27: {1:[2,3,2], 2:[2], 3:[2], 4:[2], 5:[2]},
                 28: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 29: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 30: {1:[1,1,2], 2:[1], 3:[1], 4:[2], 5:[1]},
                 31: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 32: {1:[3,3,3], 2:[3], 3:[2], 4:[3], 5:[3]},
                 33: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 34: {1:[2,2,2], 2:[2], 3:[2], 4:[2], 5:[2]},
                 35: {1:[2,2,2], 2:[3], 3:[2], 4:[3], 5:[2]},
                 36: {1:[4,3,3], 2:[4], 3:[3], 4:[4], 5:[3]},
                 37: {1:[2,2,1], 2:[2], 3:[2], 4:[3], 5:[2]},
                 38: {1:[2,3,2], 2:[3], 3:[2], 4:[3], 5:[3]},
                 39: {1:[3,3,3], 2:[3], 3:[4], 4:[3], 5:[2]},
                 40: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 41: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 42: {1:[1,2,1], 2:[2], 3:[1], 4:[1], 5:[1]},
                 43: {1:[2,3,2], 2:[2], 3:[2], 4:[2], 5:[2]},
                 44: {1:[1,2,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 45: {1:[2,2,2], 2:[2], 3:[2], 4:[2], 5:[2]}
                 }
    return responses
"""

"""
Function: random_initialization()
    Alternative initialization # 1
    Similar to initialize() above, except choose one initial class for each
    item, weighted in proportion to the counts
Input:
    counts: counts of the number of times each response was received
        by each user from each item: [items x users x classes]
Returns:
    item_classes: matrix of estimates of true item classes:
        [items x responses]
"""
def random_initialization(counts):
    [nItems, nUsers, nClasses] = np.shape(counts)

    response_sums = np.sum(counts,1)

    # create an empty array
    item_classes = np.zeros([nItems, nClasses])

    # for each item, choose a random initial class, weighted in proportion
    # to the counts from all users
    for p in range(nItems):
        average = response_sums[p,:] / np.sum(response_sums[p,:],dtype=float)
        item_classes[p,np.random.choice(np.arange(nClasses), p=average)] = 1

    return item_classes


"""
Function: majority_voting()
    Alternative initialization # 2
    An alternative way to initialize assignment of items to classes
    i.e Get initial estimates for the true item classes using majority voting
    This is not in the original paper, but could be considered
Input:
    counts: Counts of the number of times each response was received
        by each user from each item: [items x users x classes]
Returns:
    item_classes: matrix of initial estimates of true item classes:
        [items x responses]
"""
def majority_voting(counts):
    [nItems, nUsers, nClasses] = np.shape(counts)
    # sum over users
    response_sums = np.sum(counts,1)

    # create an empty array
    item_classes = np.zeros([nItems, nClasses])

    # take the most frequent class for each item
    for p in range(nItems):
        indices = np.argwhere(response_sums[p,:] == np.max(response_sums[p,:]))
        # in the case of ties, take the lowest valued label (could be randomized)
        item_classes[p, np.min(indices)] = 1

    return item_classes


if __name__ == '__main__':
    main()
