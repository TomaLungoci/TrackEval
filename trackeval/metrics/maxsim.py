
from ._base_metric import _BaseMetric
from .. import _timing

import numpy as np
from scipy.optimize import linear_sum_assignment

class MaxSim(_BaseMetric):
    """Implements the MaxSim metric, which evaluates the maximum similarity between tracked and ground truth objects."""

    def __init__(self):
        super().__init__()
        # Define which fields are floats and will be included in the summary
        self.float_fields = ['MaxSim', 'MaxSim(0)']
        self.fields = self.float_fields
        self.summary_fields = self.float_fields

    @_timing.time
    def eval_sequence(self, data):
        """Evaluate the MaxSim metric for a single sequence."""
        num_gt_ids = data['num_gt_ids']
        num_tracker_ids = data['num_tracker_ids']
        
        # Initialize an array to store maximum similarity scores between ground truth and tracked objects
        max_similarity_scores = np.zeros((num_gt_ids, num_tracker_ids), dtype=np.float32)

        # Iterate through all frames in the sequence
        for t in range(len(data['gt_ids'])):
            gt_ids_t = data['gt_ids'][t]
            tracker_ids_t = data['tracker_ids'][t]
            similarity_scores = data['similarity_scores'][t]  # Similarity matrix between ground truth and tracker

            if len(gt_ids_t) > 0 and len(tracker_ids_t) > 0:
                # For each gt_id, find the maximum similarity score across all tracker_ids
                max_sim_per_gt = np.max(similarity_scores, axis=1)

                # Update max_similarity_scores matrix
                for i, gt_id in enumerate(gt_ids_t):
                    for j, tracker_id in enumerate(tracker_ids_t):
                        max_similarity_scores[gt_id, tracker_id] += max_sim_per_gt[i]

        # Calculate the final metric result
        res = {'MaxSim': np.mean(max_similarity_scores) if np.any(max_similarity_scores > 0) else 0.0}
        res['MaxSim(0)'] = res['MaxSim']  # For example, the first value could be the same as the overall MaxSim

        return res

    def combine_sequences(self, all_res):
        """Combine results from multiple sequences."""
        combined_res = {}

        # Gather all MaxSim values across sequences
        max_sim_values = [res['MaxSim'] for res in all_res.values() if 'MaxSim' in res]

        if len(max_sim_values) > 0:
            combined_res['MaxSim'] = np.mean(max_sim_values)
        else:
            combined_res['MaxSim'] = 0.0

        combined_res['MaxSim(0)'] = combined_res['MaxSim']
        
        return combined_res

    def combine_classes_class_averaged(self, all_res):
        """Combine results by averaging across all classes."""
        combined_res = {}

        # Average MaxSim values across classes
        class_averaged_maxsim = [res['MaxSim'] for res in all_res.values() if 'MaxSim' in res and res['MaxSim'] > 0]

        combined_res['MaxSim'] = np.mean(class_averaged_maxsim) if len(class_averaged_maxsim) > 0 else 0.0
        combined_res['MaxSim(0)'] = combined_res['MaxSim']

        return combined_res

    def combine_classes_det_averaged(self, all_res):
        """Combine results by averaging across detections."""
        combined_res = {}

        # Average MaxSim values across all detections
        det_averaged_maxsim = [res['MaxSim'] for res in all_res.values() if 'MaxSim' in res and res['MaxSim'] > 0]

        combined_res['MaxSim'] = np.mean(det_averaged_maxsim) if len(det_averaged_maxsim) > 0 else 0.0
        combined_res['MaxSim(0)'] = combined_res['MaxSim']

        return combined_res