"""Base anchor functions"""
from __future__ import print_function
import numpy as np
import copy
import collections


def matrix_subset(matrix, n_samples):
    if matrix.shape[0] == 0:
        return matrix
    n_samples = min(matrix.shape[0], n_samples)
    return matrix[np.random.choice(matrix.shape[0], n_samples, replace=False)]


class AnchorBaseBeam(object):
    def __init__(self):
        pass

    @staticmethod
    def kl_bernoulli(p, q):
        """
        Compute KL-divergence
        """
        p = min(0.9999999999999999, max(0.0000001, p))
        q = min(0.9999999999999999, max(0.0000001, q))
        return (p * np.log(float(p) / q) + (1 - p) *
                np.log(float(1 - p) / (1 - q)))

    @staticmethod
    def dup_bernoulli(p, level):
        """
        Check dlow_bernoulli for documentation.
        """
        lm = p
        um = min(min(1, p + np.sqrt(level / 2.)), 1)
        for j in range(1, 17):
            qm = (um + lm) / 2.
    #         print 'lm', lm, 'qm', qm, kl_bernoulli(p, qm)
            if AnchorBaseBeam.kl_bernoulli(p, qm) > level:
                um = qm
            else:
                lm = qm
        return um

    @staticmethod
    def dlow_bernoulli(p, level):
        """
        Function:
            update lower (and potentially upper) precision bound depending on KL divergence
        Inputs:
            p : probability of label perturbed sampled data = label data to be explained (precision) --> assigned to um
            level : beta (default = appr. 3) / (n_obs in data)
        
        Returns:
            lm : lower bound of "mean label" input
        
        Questions:
            why hardcoded sampling 17x?
        """
        um = p # upper
        lm = max(min(1, p - np.sqrt(level / 2.)), 0) # lower
        for j in range(1, 17): # WHY SAMPLING 17x?
            qm = (um + lm) / 2. # mean
    #         print 'lm', lm, 'qm', qm, kl_bernoulli(p, qm)
            if AnchorBaseBeam.kl_bernoulli(p, qm) > level: # if KL-divergence > threshold level
                lm = qm
            else:
                um = qm
        return lm

    @staticmethod
    def compute_beta(n_features, t, delta):
        """
        DID NOT LOOK AT THE REASON FOR THE MAGIC NUMBERS YET
        """
        alpha = 1.1
        k = 405.5
        temp = np.log(k * n_features * (t ** alpha) / delta)
        return temp + np.log(temp)
    
    @staticmethod
    def lucb(sample_fns, initial_stats, epsilon, delta, batch_size, top_n,
             verbose=False, verbose_every=1):
        """
        Inputs:
            top_n = beam size
        
        Returns:
            indices of nb (beam width) best anchor options
        """
        # initialize 
        # initial_stats must have n_samples, positive
        
        # len(sample_fns) = nb of tuples and ...
        # ... tuples = [x for x in tuples if state['t_coverage'][x] > best_coverage]
        # so nb of tuples equal to nb of tuples with bigger coverage than current best coverage
        # -> b/c increasing coverage when adding new feature to anchors
        n_features = len(sample_fns)
        # turn n_samples and positives lists in arrays
        n_samples = np.array(initial_stats['n_samples'])
        positives = np.array(initial_stats['positives'])
        # upper and lower bound list for each option
        ub = np.zeros(n_samples.shape)
        lb = np.zeros(n_samples.shape)
        for f in np.where(n_samples == 0)[0]: # set min samples to 1
            n_samples[f] += 1
            positives[f] += sample_fns[f](1) # add labels.sum() for the sample
        if n_features == top_n: # would return all options b/c of beam search width
            return range(n_features)
        means = positives / n_samples # probability of prediction to be explained data = prediction sample
        t = 1

        def update_bounds(t):
            """
            Update lower and upper confidence bounds for each option.
            """
            sorted_means = np.argsort(means) # sort by precision (lowest probability first)
            
            # get beta - not sure what it does exactly
            beta = AnchorBaseBeam.compute_beta(n_features, t, delta)
            
            # J = nb of beam width options for anchors with highest precision
            # not_J = the rest
            J = sorted_means[-top_n:]
            not_J = sorted_means[:-top_n]
            
            for f in not_J: # update upper bound for lowest means
                ub[f] = AnchorBaseBeam.dup_bernoulli(means[f], beta / n_samples[f])
                
            for f in J: # update lower bound for highest means
                lb[f] = AnchorBaseBeam.dlow_bernoulli(means[f], beta / n_samples[f])
                
            ut = not_J[np.argmax(ub[not_J])] # ut = max for upper bound for lowest means
            lt = J[np.argmin(lb[J])] # lt = min for lower bound for highest means
            return ut, lt
        
        # keep updating upper/lower bounds until convergence incl. beam width
        # -> B > epsilon follows eq.5 from paper
        ut, lt = update_bounds(t)
        B = ub[ut] - lb[lt]
        verbose_count = 0
        while B > epsilon:
            verbose_count += 1
            if verbose and verbose_count % verbose_every == 0:
                print('Best: %d (mean:%.10f, n: %d, lb:%.4f)' %
                      (lt, means[lt], n_samples[lt], lb[lt]), end=' ')
                print('Worst: %d (mean:%.4f, n: %d, ub:%.4f)' %
                      (ut, means[ut], n_samples[ut], ub[ut]), end=' ')
                print('B = %.2f' % B)
            n_samples[ut] += batch_size
            positives[ut] += sample_fns[ut](batch_size)
            means[ut] = positives[ut] / n_samples[ut]
            n_samples[lt] += batch_size
            positives[lt] += sample_fns[lt](batch_size)
            means[lt] = positives[lt] / n_samples[lt]
            t += 1
            ut, lt = update_bounds(t)
            B = ub[ut] - lb[lt]
        sorted_means = np.argsort(means)
        return sorted_means[-top_n:]

    @staticmethod
    def make_tuples(previous_best, state):
        """
        Update tuple list with all info of possible anchors etc.
        """
        # alters state, computes support for new tuples
        normalize_tuple = lambda x: tuple(sorted(set(x)))  # noqa
        all_features = range(state['n_features'])
        coverage_data = state['coverage_data']
        current_idx = state['current_idx']
        data = state['data'][:current_idx]
        labels = state['labels'][:current_idx]
        
        # initially, every feature seperately is an anchor
        if len(previous_best) == 0:
            tuples = [(x, ) for x in all_features]
            for x in tuples:
                pres = data[:, x[0]].nonzero()[0]
                # NEW
                state['t_idx'][x] = set(pres)
                state['t_nsamples'][x] = float(len(pres))
                state['t_positives'][x] = float(labels[pres].sum())
                state['t_order'][x].append(x[0])
                # NEW
                state['t_coverage_idx'][x] = set(coverage_data[:, x[0]].nonzero()[0])
                state['t_coverage'][x] = (float(len(state['t_coverage_idx'][x])) /
                                          coverage_data.shape[0])
            return tuples
        
        # create new anchors: add a feature to every anchor in current best
        new_tuples = set()
        for f in all_features:
            for t in previous_best:
                new_t = normalize_tuple(t + (f, ))
                if len(new_t) != len(t) + 1:
                    continue
                if new_t not in new_tuples:
                    new_tuples.add(new_t)
                    state['t_order'][new_t] = copy.deepcopy(state['t_order'][t])
                    state['t_order'][new_t].append(f)
                    state['t_coverage_idx'][new_t] = (
                        state['t_coverage_idx'][t].intersection(
                            state['t_coverage_idx'][(f,)]))
                    state['t_coverage'][new_t] = (
                        float(len(state['t_coverage_idx'][new_t])) /
                        coverage_data.shape[0])
                    t_idx = np.array(list(state['t_idx'][t]))
                    t_data = state['data'][t_idx]
                    present = np.where(t_data[:, f] == 1)[0]
                    state['t_idx'][new_t] = set(t_idx[present])
                    idx_list = list(state['t_idx'][new_t])
                    state['t_nsamples'][new_t] = float(len(idx_list))
                    state['t_positives'][new_t] = np.sum(
                        state['labels'][idx_list])
        return list(new_tuples)

    @staticmethod
    def get_sample_fns(sample_fn, tuples, state):
        """
        Input:
            sample_fn : returns
                raw_data : sampled data with feature values in same discretized bin / same categorical values as
                           observation to be explained
                data : raw_data in categorical form using mapping
                labels : whether the predictions on the perturbed data is the same as the (proxy, through prediction)
                         label of the data to be explained!
        Returns:
            list of sample functions
            each sample function returns labels.sum() --> sum of where perturbed data prediction is same as data to be explained
            
        Question: if only labels.sum() returned, why the need for everything else? updated outside function?
        """
        # each sample fn returns number of positives
        sample_fns = []
        def complete_sample_fn(t, n):
            raw_data, data, labels = sample_fn(list(t), n)
            current_idx = state['current_idx']
            # idxs = range(state['data'].shape[0], state['data'].shape[0] + n)
            idxs = range(current_idx, current_idx + n)
            state['t_idx'][t].update(idxs)
            state['t_nsamples'][t] += n
            state['t_positives'][t] += labels.sum()
            state['data'][idxs] = data
            state['raw_data'][idxs] = raw_data
            state['labels'][idxs] = labels
            state['current_idx'] += n
            if state['current_idx'] >= state['data'].shape[0] - max(1000, n):
                prealloc_size = state['prealloc_size']
                current_idx = data.shape[0]
                state['data'] = np.vstack((state['data'],
                                           np.zeros((prealloc_size, data.shape[1]), data.dtype)))
                state['raw_data'] = np.vstack((state['raw_data'],
                                               np.zeros((prealloc_size, raw_data.shape[1]),
                                                        raw_data.dtype)))
                state['labels'] = np.hstack((state['labels'],
                                             np.zeros(prealloc_size, labels.dtype)))
            # This can be really slow
            # state['data'] = np.vstack((state['data'], data))
            # state['raw_data'] = np.vstack((state['raw_data'], raw_data))
            # state['labels'] = np.hstack((state['labels'], labels))
            return labels.sum()
        
        for t in tuples:
            sample_fns.append(lambda n, t=t: complete_sample_fn(t, n))
            
        return sample_fns


    @staticmethod
    def get_initial_statistics(tuples, state):
        """
        For each tuple (potential anchor), return nb of samples used and where preds=desired labels.
        """
        stats = {
            'n_samples': [],
            'positives': []
        }
        for t in tuples:
            stats['n_samples'].append(state['t_nsamples'][t])
            stats['positives'].append(state['t_positives'][t])
        return stats

    @staticmethod
    def get_anchor_from_tuple(t, state):
        # TODO: This is wrong, some of the intermediate anchors may not exist.
        anchor = {'feature': [], 'mean': [], 'precision': [],
                  'coverage': [], 'examples': [], 'all_precision': 0}
        anchor['num_preds'] = state['data'].shape[0]
        normalize_tuple = lambda x: tuple(sorted(set(x)))  # noqa
        current_t = tuple()
        for f in state['t_order'][t]:
            current_t = normalize_tuple(current_t + (f,))

            mean = (state['t_positives'][current_t] /
                    state['t_nsamples'][current_t])
            anchor['feature'].append(f)
            anchor['mean'].append(mean)
            anchor['precision'].append(mean)
            anchor['coverage'].append(state['t_coverage'][current_t])
            raw_idx = list(state['t_idx'][current_t])
            raw_data = state['raw_data'][raw_idx]
            covered_true = (
                state['raw_data'][raw_idx][state['labels'][raw_idx] == 1])
            covered_false = (
                state['raw_data'][raw_idx][state['labels'][raw_idx] == 0])
            exs = {}
            exs['covered'] = matrix_subset(raw_data, 10)
            exs['covered_true'] = matrix_subset(covered_true, 10)
            exs['covered_false'] = matrix_subset(covered_false, 10)
            exs['uncovered_true'] = np.array([])
            exs['uncovered_false'] = np.array([])
            anchor['examples'].append(exs)
        return anchor

    @staticmethod
    def anchor_beam(sample_fn, delta=0.05, epsilon=0.1, batch_size=10,
                    min_shared_samples=0, desired_confidence=1, beam_size=1,
                    verbose=False, epsilon_stop=0.05, min_samples_start=0,
                    max_anchor_size=None, verbose_every=1,
                    stop_on_first=False, coverage_samples=10000):
        
        """
        Function called from explain_instance.
        
        Inputs:
            sample_fn : function that samples data from training or validation set
                        returns: raw_data : sampled data with feature values in same discretized
                                            bin / same categorical values as observation to be explained
                                 data : raw_data in categorical form using mapping for ordinal features -> extra columns
                                 labels : whether perturbed data labels are same as label of obs to be explained
        
        sample_fn(present, num_samples, compute_labels=True, validation=True)
        
        Returns:
            best anchors
        """
        
        # initiate empty anchor dict
        anchor = {'feature': [], 'mean': [], 'precision': [],
                  'coverage': [], 'examples': [], 'all_precision': 0}
        
        # sample # coverage_samples data in categorical form, not interested in raw_data or labels
        # IMPORTANT: b/c first argument = [] -> conditions_eq/leq/geq = {} 
        # -> true random sampling of training (or validation) set
        _, coverage_data, _ = sample_fn([], coverage_samples, compute_labels=False)
        
        # sample by default 1 more random value
        raw_data, data, labels = sample_fn([], max(1, min_samples_start))
        mean = labels.mean()
        beta = np.log(1. / delta)
        
        # inputs: mean label = probability of label sampled data = label data to be explained
        #         level = beta / (n_obs in data)
        # returns lower probability bound of "mean label" input
        lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / data.shape[0])
        
        # desired_confidence typically 0.95 so "mean preds=desired label > 0.95"
        # -> see eq.3 in paper: prec(A)>tau --> precision constraint
        # lower probability bound of pred=desired label < 0.95-eps (eps eg 0.1-0.15)
        # -> see eq.5 in paper: prec(A) > prec(A*) - eps where prec(A*) >= tau for sufficient confidence of mean
        # while loop: keep adding data until lb is high enough
        # -> see end of "bottom-up construction of anchors" in paper: prec_lb(A) < tau but ...
        # ... prec_ub(A) > tau -> sample from D(.|A) until confident A prec_lb(A) > tau or < tau
        # iteratively update lower probability bound with batches of newly sampled data
        while mean > desired_confidence and lb < desired_confidence - epsilon:
            nraw_data, ndata, nlabels = sample_fn([], batch_size)
            data = np.vstack((data, ndata))
            raw_data = np.vstack((raw_data, nraw_data))
            labels = np.hstack((labels, nlabels))
            mean = labels.mean()
            lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / data.shape[0])
            
        # -> see end of "bottom-up construction of anchors" in paper: ... until confident A prec_lb(A) > tau ...
        if lb > desired_confidence:
            anchor['num_preds'] = data.shape[0]
            anchor['all_precision'] = mean
            return anchor
        
        # initialize variables
        prealloc_size = batch_size * 10000
        current_idx = data.shape[0]
        data = np.vstack((data, 
                          np.zeros((prealloc_size, data.shape[1]),data.dtype)))
        raw_data = np.vstack((raw_data, 
                              np.zeros((prealloc_size, raw_data.shape[1]),raw_data.dtype)))
        labels = np.hstack((labels, 
                            np.zeros(prealloc_size, labels.dtype)))
        n_features = data.shape[1]
        state = {'t_idx': collections.defaultdict(lambda: set()),
                 't_nsamples': collections.defaultdict(lambda: 0.),
                 't_positives': collections.defaultdict(lambda: 0.),
                 'data': data,
                 'prealloc_size': prealloc_size,
                 'raw_data': raw_data,
                 'labels': labels,
                 'current_idx': current_idx,
                 'n_features': n_features,
                 't_coverage_idx': collections.defaultdict(lambda: set()),
                 't_coverage': collections.defaultdict(lambda: 0.),
                 'coverage_data': coverage_data,
                 't_order': collections.defaultdict(lambda: list())
                 }
        current_size = 1
        best_of_size = {0: []}
        best_coverage = -1
        best_tuple = ()
        t = 1
        if max_anchor_size is None:
            max_anchor_size = n_features
        
        # find best anchor using beam search until max anchor size
        while current_size <= max_anchor_size:
            
            # create new potential anchors by adding features to current best anchors
            tuples = AnchorBaseBeam.make_tuples(best_of_size[current_size - 1], state)
            
            # need to max coverage with P(prec(A)>tau)>1-delta as constraint (eq.4 in paper)
            # so keep tuples in list that are better than current best coverage
            tuples = [x for x in tuples if state['t_coverage'][x] > best_coverage]
            
            # coverage can only increase, so if no better coverage found with added features -> break
            if len(tuples) == 0:
                break
            
            # build sample functions for each tuple in tuples list
            # these sample functions would sample randomly for all features except for the
            # features in the anchors where it samples from the category or bin (for discretized numerical
            # features) of the feature value in the to be explained observation
            sample_fns = AnchorBaseBeam.get_sample_fns(sample_fn, tuples, state)
            
            # initial n_samples and positives stats
            initial_stats = AnchorBaseBeam.get_initial_statistics(tuples, state)
            
            # apply LUCB and get anchor options back in the form of tuple indices
            # print tuples, beam_size
            chosen_tuples = AnchorBaseBeam.lucb(sample_fns, 
                                                initial_stats, 
                                                epsilon, delta, 
                                                batch_size,
                                                min(beam_size, len(tuples)),
                                                verbose=verbose, 
                                                verbose_every=verbose_every)
            
            # update with best tuples for each anchor size (nb of features in each anchor)
            best_of_size[current_size] = [tuples[x] for x in chosen_tuples]
            if verbose:
                print('Best of size ', current_size, ':')
            # print state['data'].shape[0]
            
            # for each potential anchor:
            # update precision, lower and upper bounds until precision constraints are met
            # update best anchor if coverage is larger than current best coverage
            stop_this = False
            for i, t in zip(chosen_tuples, best_of_size[current_size]):
                
                # I can choose at most (beam_size - 1) tuples at each step,
                # and there are at most n_feature steps
                beta = np.log(1. / (delta / (1 + (beam_size - 1) * n_features)))
                # beta = np.log(1. / delta)
                
                # if state['t_nsamples'][t] == 0:
                #     mean = 1
                # else:
                
                # get precision, lower and upper bounds, and coverage for potential anchor
                mean = state['t_positives'][t] / state['t_nsamples'][t]
                lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / state['t_nsamples'][t])
                ub = AnchorBaseBeam.dup_bernoulli(mean, beta / state['t_nsamples'][t])
                coverage = state['t_coverage'][t]
                if verbose:
                    print(i, mean, lb, ub)
                
                # while: (see end of "bottom-up construction of anchors" section)
                # 1. precision > tau (eg 0.95) (eq.3) and tau - prec_lb > eps (eq.5)
                #    --> prec_lb needs to improve
                # 2. precision < tau and ub - tau >= eps
                #    --> precision needs to improve
                while ((mean >= desired_confidence and lb < desired_confidence - epsilon_stop) or
                       (mean < desired_confidence and ub >= desired_confidence + epsilon_stop)):
                    
                    # sample a batch of data, get new precision, lb and ub values
                    # print mean, lb, state['t_nsamples'][t]
                    sample_fns[i](batch_size)
                    mean = state['t_positives'][t] / state['t_nsamples'][t]
                    lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / state['t_nsamples'][t])
                    ub = AnchorBaseBeam.dup_bernoulli(mean, beta / state['t_nsamples'][t])
                
                if verbose:
                    print('%s mean = %.2f lb = %.2f ub = %.2f coverage: %.2f n: %d' % \
                          (t, mean, lb, ub, coverage, state['t_nsamples'][t]))
                    
                # if precision > tau and tau - prec_lb <= eps --> found eligible anchor
                if mean >= desired_confidence and lb > desired_confidence - epsilon_stop:
                    if verbose:
                        print('Found eligible anchor ', t, 'Coverage:',
                              coverage, 'Is best?', coverage > best_coverage)
                        
                    # coverage eligible anchor needs to be bigger than current best coverage
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_tuple = t
                        if best_coverage == 1 or stop_on_first:
                            stop_this = True
            if stop_this:
                break
            current_size += 1
        
        
        if best_tuple == ():
            # Could not find an anchor, will now choose the highest precision
            # amongst the top K from every round
            if verbose:
                print('Could not find an anchor, now doing best of each size')
            tuples = []
            for i in range(0, current_size):
                tuples.extend(best_of_size[i])
            # tuples = best_of_size[current_size - 1]
            sample_fns = AnchorBaseBeam.get_sample_fns(sample_fn, tuples, state)
            initial_stats = AnchorBaseBeam.get_initial_statistics(tuples, state)
            # print tuples, beam_size
            chosen_tuples = AnchorBaseBeam.lucb(sample_fns, initial_stats, epsilon, delta, batch_size,1, verbose=verbose)
            best_tuple = tuples[chosen_tuples[0]]
        
        # return best_tuple, state
        return AnchorBaseBeam.get_anchor_from_tuple(best_tuple, state)
