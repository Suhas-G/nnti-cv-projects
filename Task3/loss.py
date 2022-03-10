import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        # It was empirically found that scale of class contrastive loss is 100 times bigger 
        # than the scale of augmentation contrastive loss. So balancing them using the coefficient.
        self.balancing_coeff = 0.01
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def augmentation_contrastive_loss(self, z_i, z_j):
        '''Find cosine similarity of each pair from z_i and z_j.
        Coresponding pairs of z_i and z_j are considered as positives, whose distance has to be minimised.
        Hence form the numerator of the loss.
        All other pairs are considered as negatives, whose distance has to be maximised.
        And form the denominator of the loss.
        '''
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        assert similarity_matrix.size(0) == similarity_matrix.size(1)
        assert int(similarity_matrix.size(0) / 2) == z_i.size(0)
    
        sim_ij = torch.diag(similarity_matrix, similarity_matrix.size(0) // 2)
        sim_ji = torch.diag(similarity_matrix, -(similarity_matrix.size(0) // 2))
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        numerator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask[: similarity_matrix.size(0), :similarity_matrix.size(1)] * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * similarity_matrix.size(0))
        return loss

    def class_contrastive_loss(self, z_i, z_j, predicted_cls):
        '''Find cosine similarity of each pair from filtered z_i and z_j.
        All pairs that have same predicted class are considered positive, whose distance has to be minimised.
        Hence forms the numerator of the loss.
        All other pairs are considered as negatives, whose distance has to be maximised. And form the
        denominator of the loss.
        '''
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        cls_representations = torch.cat([predicted_cls, predicted_cls]).unsqueeze(0)
        cls_similarity_matrix = torch.isclose(torch.bitwise_xor(cls_representations, cls_representations.T), torch.tensor([0]).to(cls_representations.device))

        positives = (cls_similarity_matrix.float() - torch.eye(cls_similarity_matrix.size(0)).to(cls_similarity_matrix.device)) * similarity_matrix
        negatives = (~cls_similarity_matrix).float() * similarity_matrix

        numerator = torch.exp(positives / self.temperature)
        denominator = torch.exp(negatives / self.temperature)

        loss_partial = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * cls_similarity_matrix.size(0))
        return loss


            
    def forward(self, emb_i, emb_j, predicted_cls):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        loss1 = self.augmentation_contrastive_loss(z_i, z_j)
        # If predicted class is -1, it means the confidence was lesser than FixMatch threshold.
        reduced_z_i = z_i[predicted_cls != -1]
        reduced_z_j = z_j[predicted_cls != -1]

        if reduced_z_i.size(0) == 0:
            assert reduced_z_j.size(0) == 0
            loss2 = 0.0
        else:
            loss2 = self.balancing_coeff * self.class_contrastive_loss(reduced_z_i, reduced_z_j, predicted_cls[predicted_cls != -1])



        return loss1 + loss2

def get_consistency_loss(weakly_augmented_outputs, strongly_augmented_outputs, threshold):
    '''Cross entropy loss between the weakly augmented image class prediction and 
    the strongly augmented image class prediction for weakly augmented class predictions 
    that are greater than a threshold.
    If no weakly augmented class prediction is greater than the threshold, then the we take loss 
    as 0.
    '''
    pseudo_label = torch.softmax(weakly_augmented_outputs.detach(), dim=-1)
    max_probs, targets_u = torch.max(pseudo_label, dim = -1)
    mask = max_probs.ge(threshold)
    valid_targets = targets_u[mask]
    if (valid_targets.size(0) == 0):
        return torch.tensor(0).to(weakly_augmented_outputs.device)
    return F.cross_entropy(strongly_augmented_outputs[mask], valid_targets)