import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from mamba_ssm import Mamba2
import json
# Assuming you have imported Mamba2 as specified

class CombinedModel(nn.Module):
    def __init__(self, base_model):
        super(CombinedModel, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.d_model, 4)  # Classifier for 4 classes

    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x

emblist1,emblist2,labellist = [],[],[]
temp = 0
# 假设embedding1和embedding2是两个包含embedding向量的列表
with open('../datasets_p/Testdatavec.json', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        qvec = data["qfact_vec"]
        cvec = data["cfact_vec"]
        label = data["label"]
        emblist1 = emblist1 + [qvec]
        emblist2 = emblist2 + [cvec]
        labellist = labellist + [int(label)]

Temblist1,Temblist2,Tlabellist = [],[],[]
# 假设embedding1和embedding2是两个包含embedding向量的列表
with open('../datasets_p/Testdatavec.json', 'r', encoding='utf-8') as f:
    for line in f:
        Tdata = json.loads(line)
        Tqvec = Tdata["qfact_vec"]
        Tcvec = Tdata["cfact_vec"]
        Tlabel = Tdata["label"]
        Temblist1 = Temblist1 + [qvec]
        Temblist2 = Temblist2 + [cvec]
        Tlabellist = Tlabellist + [int(label)]
# Sample data preparation (replace this with your actual data)
# Assuming embeddings1 and embeddings2 are tensors of shape (num_samples, 1024)
# and labels is a tensor of shape (num_samples,)
embedding1 = torch.tensor(emblist1,device="cuda")
embedding2 = torch.tensor(emblist2,device="cuda")
labels = torch.tensor(labellist,device="cuda")  # 0表示一类，1表示另一类
Tembedding1 = torch.tensor(Temblist1,device="cuda")
Tembedding2 = torch.tensor(Temblist2,device="cuda")
Tlabels = torch.tensor(Tlabellist,device="cuda")

# Concatenate the embeddings
combined_embeddings = torch.cat((embedding1, embedding2), dim=1)

# Create a DataLoader
dataset = TensorDataset(combined_embeddings, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the base model
dim = 2048  # d_model set to 2048 to match the combined embeddings
base_model = Mamba2(
    d_model=dim,
    d_state=64,
    d_conv=4,
    expand=2,
).to("cuda")

# Wrap the base model with the classifier
model = CombinedModel(base_model).to("cuda")

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_embeddings, batch_labels in dataloader:
        batch_embeddings, batch_labels = batch_embeddings.to("cuda"), batch_labels.to("cuda")
        optimizer.zero_grad()
        outputs = model(batch_embeddings)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

print("Training completed.")
torch.save(model.state_dict(), "mamba2_model.pth")
print("Model saved.")

# Evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_embeddings, batch_labels in test_loader:
        batch_embeddings, batch_labels = batch_embeddings.to("cuda"), batch_labels.to("cuda")
        outputs = model(batch_embeddings)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {accuracy}")

# Calculate mean AUC-ROC and mean PR-AUC
all_preds_prob = []
with torch.no_grad():
    for batch_embeddings, batch_labels in test_loader:
        batch_embeddings, batch_labels = batch_embeddings.to("cuda"), batch_labels.to("cuda")
        outputs = model(batch_embeddings)
        preds_prob = torch.softmax(outputs, dim=1)
        all_preds_prob.extend(preds_prob.cpu().numpy())

all_labels_one_hot = nn.functional.one_hot(torch.tensor(all_labels), num_classes=4).numpy()

mean_auc_roc = roc_auc_score(all_labels_one_hot, all_preds_prob, average='macro', multi_class='ovo')
mean_pr_auc = average_precision_score(all_labels_one_hot, all_preds_prob, average='macro')

print(f"Mean AUC-ROC: {mean_auc_roc}")
print(f"Mean PR-AUC: {mean_pr_auc}")