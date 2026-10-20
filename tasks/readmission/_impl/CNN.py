from tasks.readmission._paths import DATA_DIR, MODEL_DIR, RAW_DIR, data_file, output_file, difficulty_file, device_name
import os
import numpy as np
import pandas as pd
import gensim
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc

# -------------------------------
# 1. 定义模型：TextCNN
# -------------------------------
class TextCNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_labels=1, embedding_matrix=None, filter_sizes=[3,4,5], num_filters=100):
        super(TextCNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_labels)

    def forward(self, input_ids, labels=None):
        embeds = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        x = embeds.permute(0, 2, 1)  # [batch_size, embed_dim, seq_len]
        conv_outputs = [torch.relu(conv(x)) for conv in self.convs]
        pooled_outputs = [torch.max(c, dim=2)[0] for c in conv_outputs]
        cat = torch.cat(pooled_outputs, dim=1)
        cat = self.dropout(cat)
        logits = self.fc(cat)
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float().view(-1, 1))
            return loss, logits
        else:
            return logits

# -------------------------------
# 2. 数据集定义
# -------------------------------
class ClinicalTextDataset(Dataset):
    def __init__(self, csv_file, word2idx, max_seq_length=512):
        self.data = pd.read_csv(csv_file)
        self.texts = self.data["TEXT"].astype(str).tolist()
        self.labels = self.data["Label"].values
        self.word2idx = word2idx
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.texts)

    def tokenize(self, text):
        return text.strip().split()

    def numericalize(self, tokens):
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]

    def pad_sequence(self, seq):
        if len(seq) < self.max_seq_length:
            seq = seq + [self.word2idx["<PAD>"]] * (self.max_seq_length - len(seq))
        else:
            seq = seq[:self.max_seq_length]
        return seq

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenize(text)
        indices = self.numericalize(tokens)
        indices = self.pad_sequence(indices)
        label = self.labels[idx]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# -------------------------------
# 3. 主函数
# -------------------------------
def main():
    # 路径和参数
    train_csv = data_file('train', str(DATA_DIR / 'dead/data40k/train.csv'))
    val_csv = data_file('val', str(DATA_DIR / 'dead/data40k/val.csv'))
    test_csv = data_file('test', str(DATA_DIR / 'dead/data40k/test.csv'))
    wv_model_path = __import__("os").environ.get("TACL_READMISSION_WORD2VEC_MODEL", str(MODEL_DIR / 'word2vec.model'))
    max_seq_length = 2048
    batch_size = 128
    num_epochs = 30
    learning_rate = 5e-4
    patience = 3

    device = torch.device(device_name("cuda:0" if torch.cuda.is_available() else "cpu"))

    # -------------------------------
    # 4. 加载词向量
    # -------------------------------
    print("加载 Word2Vec 模型...")
    wv_model = gensim.models.KeyedVectors.load(wv_model_path)
    original_vocab = wv_model.wv.index_to_key
    embed_dim = wv_model.vector_size
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    vocab_size = len(original_vocab) + 2
    embedding_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    embedding_matrix[1] = np.random.normal(scale=0.6, size=(embed_dim,))
    for i, word in enumerate(original_vocab):
        idx = i + 2
        word2idx[word] = idx
        embedding_matrix[idx] = wv_model.wv[word]
    print(f"词表大小: {vocab_size}, 词向量维度: {embed_dim}")

    # -------------------------------
    # 5. 构建 Dataset 和 DataLoader
    # -------------------------------
    train_dataset = ClinicalTextDataset(train_csv, word2idx, max_seq_length)
    val_dataset = ClinicalTextDataset(val_csv, word2idx, max_seq_length)
    test_dataset = ClinicalTextDataset(test_csv, word2idx, max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------------
    # 6. 初始化模型
    # -------------------------------
    model = TextCNNClassifier(vocab_size, embed_dim, num_labels=1, embedding_matrix=embedding_matrix)
    model = model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    # -------------------------------
    # 7. 训练与验证
    # -------------------------------
    for epoch in range(num_epochs):
        model.train()
        losses = []
        for step, (input_ids, labels) in enumerate(train_loader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            loss, logits = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1} Train Loss: {np.mean(losses):.4f}")

        # 验证
        model.eval()
        val_losses, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                loss, logits = model(input_ids, labels)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy().flatten())
        avg_val_loss = np.mean(val_losses)
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except:
            auroc = 0.0
        acc = accuracy_score(all_labels, np.array(all_probs) >= 0.5)
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        auprc = auc(recall, precision)
        rp80 = max([r for p, r in zip(precision, recall) if p >= 0.8] or [0.0])
        print(f"[验证集] Loss: {avg_val_loss:.4f}, AUROC: {auroc:.4f}, Acc: {acc:.4f}, AUPRC: {auprc:.4f}, RP80: {rp80:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_textcnn_model.pth")
            print("✅ 模型已保存")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("⏹️ 验证损失无提升，提前停止训练")
                break

    # -------------------------------
    # 8. 测试
    # -------------------------------
    print("🔍 测试阶段加载最佳模型...")
    model.load_state_dict(torch.load("best_textcnn_model.pth"))
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for input_ids, labels in test_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            loss, logits = model(input_ids, labels)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy().flatten())
    test_auroc = roc_auc_score(all_labels, all_probs)
    test_acc = accuracy_score(all_labels, np.array(all_probs) >= 0.5)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    test_auprc = auc(recall, precision)
    test_rp80 = max([r for p, r in zip(precision, recall) if p >= 0.8] or [0.0])
    print(f"[测试集] AUROC: {test_auroc:.4f}, Accuracy: {test_acc:.4f}, AUPRC: {test_auprc:.4f}, RP80: {test_rp80:.4f}")

if __name__ == "__main__":
    main()
