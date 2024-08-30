import torch
import torch.nn as nn
import torch.nn.functional as F

class LanguageAndVisionConcat(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        loss_fn,
        language_module,
        vision_module,
        text_feature_dim,
        vision_feature_dim,
        fusion_output_size,
        dropout_p,
    ):
        super(LanguageAndVisionConcat, self).__init__()
        self.language_module = language_module
        self.vision_module = vision_module
        self.fusion = torch.nn.Linear(
            in_features=(text_feature_dim + vision_feature_dim),
            out_features=fusion_output_size,
        )
        self.fc = torch.nn.Linear(
            in_features=fusion_output_size, out_features=num_classes
        )
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)
        self.batchNorm = torch.nn.BatchNorm1d(text_feature_dim + vision_feature_dim)

    def forward(self, text, image, label=None):
        # text_features = torch.nn.functional.relu(self.language_module(text))
        # image_features = torch.nn.functional.relu(self.vision_module(image))
        # combined = torch.cat([text_features, image_features], dim=1)
        # fused = self.dropout(torch.nn.functional.relu(self.fusion(combined)))
        # logits = self.fc(fused)
        # pred = torch.nn.functional.softmax(logits, dim=1)
        # loss = self.loss_fn(pred, label) if label is not None else label
        # pred = torch.argmax(pred, dim=1)
        # return (pred, loss)
        text_features = self.language_module(text)
        image_features = self.vision_module(image)
        combined = torch.cat([text_features, image_features], dim=1)
        combined = torch.nn.functional.relu(self.batchNorm(combined))
        fused = self.dropout(torch.nn.functional.relu(self.fusion(combined)))
        logits = self.fc(fused)
        pred = torch.nn.functional.softmax(logits, dim=1)
        loss = self.loss_fn(pred, label) if label is not None else label
        pred = torch.argmax(pred, dim=1)
        return (pred, loss)


class LanguageAndVisionAttentionOptA(nn.Module):
    def __init__(
        self,
        num_classes,
        loss_fn,
        language_module,
        vision_module,
        text_feature_dim,
        vision_feature_dim,
        fusion_output_size,
        dropout_p,
    ):
        super(LanguageAndVisionAttentionOptA, self).__init__()
        self.language_module = language_module
        self.vision_module = vision_module

        # Transformation matrices
        self.weight_transformer_text = nn.Parameter(torch.randn(text_feature_dim, fusion_output_size))
        self.weight_transformer_vision = nn.Parameter(torch.randn(vision_feature_dim, fusion_output_size))
        self.weight_a = nn.Parameter(torch.randn(fusion_output_size, 1))

        self.fusion = nn.Linear(in_features=fusion_output_size, out_features=fusion_output_size)
        self.fc = nn.Linear(in_features=fusion_output_size, out_features=num_classes)
        self.loss_fn = loss_fn
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, text, image, label=None):
        # Extract features
        text_features = F.relu(self.language_module(text))
        image_features = F.relu(self.vision_module(image))

        # Transform features
        w_text = torch.mm(text_features, self.weight_transformer_vision)
        w_image = torch.mm(image_features, self.weight_transformer_text)

        # Normalize features to have a meaningful range
        w_text = F.layer_norm(w_text, w_text.shape[1:])
        w_image = F.layer_norm(w_image, w_image.shape[1:])

        # Apply non-linearity to transformed features
        w_text = F.tanh(w_text)
        w_image = F.tanh(w_image)

        # Compute attention scores
        g_text = torch.mm(w_text, self.weight_a)
        g_image = torch.mm(w_image, self.weight_a)

        # Concatenate attention scores and apply activations
        alpha = torch.cat([g_text, g_image], dim=1)
        alpha = F.leaky_relu(alpha, 0.02)
        alpha = F.softmax(alpha, dim=-1)

        # Combine the weighted features
        stack_tensors = [w_text, w_image]
        combined = torch.stack(stack_tensors, dim=1)
        outputs_w_attention = alpha[:, :, None] * combined
        combined_feats = outputs_w_attention.sum(dim=1)

        # Fusion and classification
        fused = self.dropout(F.relu(self.fusion(combined_feats)))
        logits = self.fc(fused)
        pred_probs = F.softmax(logits, dim=1)

        # Calculate loss if label is provided
        loss = self.loss_fn(pred_probs, label) if label is not None else None

        # Predictions
        preds = torch.argmax(pred_probs, dim=1)
        return (preds, loss)

class LanguageAndVisionCrossAttention(nn.Module):
    def __init__(
        self,
        num_classes,
        loss_fn,
        language_module,
        vision_module,
        text_feature_dim,
        vision_feature_dim,
        fusion_output_size,
        dropout_p,
    ):
        super(LanguageAndVisionCrossAttention, self).__init__()
        self.language_module = language_module
        self.vision_module = vision_module

        # Transformation matrices for cross attention
        self.weight_transformer_text = nn.Parameter(torch.randn(text_feature_dim, fusion_output_size) * 0.01)
        self.weight_transformer_vision = nn.Parameter(torch.randn(vision_feature_dim, fusion_output_size) * 0.01)
        self.weight_cross_text = nn.Parameter(torch.randn(fusion_output_size, fusion_output_size) * 0.01)
        self.weight_cross_vision = nn.Parameter(torch.randn(fusion_output_size, fusion_output_size) * 0.01)
        self.weight_a = nn.Parameter(torch.randn(fusion_output_size, 1) * 0.01)

        self.fusion = nn.Linear(in_features=fusion_output_size, out_features=fusion_output_size)
        self.fc = nn.Linear(in_features=fusion_output_size, out_features=num_classes)
        self.loss_fn = loss_fn
        self.dropout = nn.Dropout(dropout_p)
        # print("self.dropout", self.dropout)

    def forward(self, text, image, label=None):
        # Extract features
        text_features = F.relu(self.language_module(text))
        image_features = F.relu(self.vision_module(image))

        # Transform features
        w_text = torch.mm(text_features, self.weight_transformer_text)
        w_image = torch.mm(image_features, self.weight_transformer_vision)

        # Normalize features to have a meaningful range
        w_text = F.layer_norm(w_text, w_text.shape[1:])
        w_image = F.layer_norm(w_image, w_image.shape[1:])

        # Apply cross attention
        cross_text = torch.mm(w_image, self.weight_cross_text)
        cross_vision = torch.mm(w_text, self.weight_cross_vision)

        # Apply non-linearity to transformed features
        cross_text = F.tanh(cross_text)
        cross_vision = F.tanh(cross_vision)

        # Compute attention scores
        g_text = torch.mm(cross_text, self.weight_a)
        g_vision = torch.mm(cross_vision, self.weight_a)

        # Concatenate attention scores and apply activations
        alpha = torch.cat([g_text, g_vision], dim=1)
        alpha = F.leaky_relu(alpha, 0.02)
        alpha = F.softmax(alpha, dim=-1)

        # Combine the weighted features
        stack_tensors = [cross_text, cross_vision]
        combined = torch.stack(stack_tensors, dim=1)
        outputs_w_attention = alpha[:, :, None] * combined
        combined_feats = outputs_w_attention.sum(dim=1)

        # Fusion and classification
        fused = self.dropout(self.fusion(combined_feats))
        logits = self.fc(fused)
        pred_probs = F.softmax(logits, dim=1)

        # Calculate loss if label is provided
        loss = self.loss_fn(pred_probs, label) if label is not None else None

        # Predictions
        preds = torch.argmax(pred_probs, dim=1)
        return (preds, loss)

class CrossModalAttention(torch.nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim):
        super(CrossModalAttention, self).__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim

        self.text_to_hidden = nn.Linear(text_dim, hidden_dim)
        self.image_to_hidden = nn.Linear(image_dim, hidden_dim)
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, text_embeddings, image_embeddings):
        # Transform both text and image embeddings to the same hidden space
        text_hidden = self.text_to_hidden(text_embeddings)  # Shape: (batch_size, text_len, hidden_dim)
        image_hidden = self.image_to_hidden(image_embeddings)  # Shape: (batch_size, image_len, hidden_dim)
        # Apply attention mechanism
        text_attention_scores = self.attention_weights(F.relu(text_hidden))  # Shape: (batch_size, text_len)
        image_attention_scores = self.attention_weights(F.relu(image_hidden))  # Shape: (batch_size, image_len)

        # Compute attention weights
        text_attention_weights = F.softmax(text_attention_scores, dim=1)  # Shape: (batch_size, text_len)
        image_attention_weights = F.softmax(image_attention_scores, dim=1)  # Shape: (batch_size, image_len)

        # Apply attention weights
        text_context = torch.sum(text_hidden * text_attention_weights.unsqueeze(-1), dim=1)  # Shape: (batch_size, hidden_dim)
        image_context = torch.sum(image_hidden * image_attention_weights.unsqueeze(-1), dim=1)  # Shape: (batch_size, hidden_dim)

        # Concatenate contexts from both modalities
        combined_context = torch.cat((text_context, image_context), dim=1)  # Shape: (batch_size, 2 * hidden_dim)

        return combined_context

class ClassifierHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ClassifierHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        pred_probs = F.softmax(x, dim=1)
        return pred_probs

class MultimodalClassifier(nn.Module):
    def __init__(self, language_module, vision_module, text_dim, image_dim, hidden_dim, classifier_hidden_dim, loss_fn):
        super(MultimodalClassifier, self).__init__()
        self.cross_modal_attention = CrossModalAttention(text_dim, image_dim, hidden_dim)
        self.classifier_head = ClassifierHead(hidden_dim * 2, classifier_hidden_dim)
        self.loss_fn = loss_fn
        self.text_module = language_module
        self.vision_module = vision_module
    def forward(self, text, image, label):
        text_embeddings = self.text_module(text)
        image_embeddings = self.vision_module(image)
        fused_embeddings = self.cross_modal_attention(text_embeddings, image_embeddings)
        pred_probs = self.classifier_head(fused_embeddings)
        preds = torch.argmax(pred_probs, dim=1)
        loss = self.loss_fn(pred_probs, label) if label is not None else None
        return (preds, loss)


class Clip(nn.Module):

    def __init__(
        self,
        text_module,
        vision_module,
        text_feature_dim,
        vision_feature_dim,
        num_classes,
        loss_fn,
        temperature=1,
        dropout_p=0.2,
    ):
        super(Clip, self).__init__()

        self.image_projection = vision_module
        self.text_projection = text_module
        self.classifier = nn.Sequential(
            nn.Linear(text_feature_dim + vision_feature_dim, 100),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(100, num_classes),
        )
        self.loss_fn = loss_fn
        self.temperature = temperature

    def forward(self, text, image, label=None):
        text_features = self.text_projection(text)
        image_features = self.image_projection(image)

        # compute similarities and get contrastive loss
        similarity_logits = (text_features @ image_features.T) / self.temperature
        image_similarity = image_features @ image_features.T
        text_similarity = text_features @ text_features.T
        targets = F.softmax(
            image_similarity + text_similarity / (2 * self.temperature), dim=1
        )
        text_loss = F.cross_entropy(similarity_logits, targets)
        image_loss = F.cross_entropy(similarity_logits.T, targets.T)

        combined_features = torch.cat((text_features, image_features), dim=1)
        logits = self.classifier(combined_features)
        pred = torch.nn.functional.softmax(logits, dim=1)
        classification_loss = self.loss_fn(pred, label) if label is not None else label
        # combine contrastive loss with classification loss
        loss = classification_loss * (text_loss + image_loss)
        pred = torch.argmax(pred, dim=1)

        return (pred, loss)
