import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



# PLOT FUNCTION
def plot_metrics(train_acc, test_acc, clean_probs, noisy_probs):
    """
    Plot the metrics during training: train/test accuracies and softmax probabilities.
    """
    # Plot Train and Test 
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(list(range(1, len(train_acc)+1)), train_acc, label='Train Accuracy')
    plt.plot(list(range(1, len(test_acc)+1)), test_acc, label='Test Accuracy')
    plt.xscale('log')
    plt.xlabel('Iteration (log scale)')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy during Training')
    plt.legend()

    # Plot Softmax Probabilities for Clean and Noisy Samples
    plt.subplot(1, 2, 2)
    plt.plot(list(range(1, len(clean_probs)+1)), clean_probs, label='Clean Sample Probabilities')
    plt.plot(list(range(1, len(noisy_probs)+1)), noisy_probs, label='Noisy Sample Probabilities')
    plt.xscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Softmax Probability')
    plt.title('Softmax Probability of Signal Token')
    plt.legend()

    plt.tight_layout()
    plt.show()


# LOSS AND ACCURACY FUNCTIONS
# Loss function
def logistic_loss_fn(logits, labels):
    return torch.mean(torch.log(1.0 + torch.exp(-labels * logits)))

# Accuracy score
def accuracy_score(logits, labels):
    preds = torch.sign(logits)  # get +1 or -1
    correct = (preds == labels).sum().item()
    return correct / len(labels)


# TRAINING WITH GRADIENT DESCENT FUNCTION
def train_with_gradient_descent(model, 
                                optimizer, 
                                X_train, 
                                y_train, 
                                X_test, 
                                y_test, 
                                clean_indices_train,
                                noisy_indices_train,
                                log_every=50, 
                                num_steps=150,
                                beta=0.025):

    # Initialize lists and step
    train_accs, test_accs, clean_proba, noisy_proba, steps_list = [0.01], [0.01], [0.5], [0.5], []
    cur_step = 0

    # Descent
    for epoch in range(num_steps):
        pred = model(X_train)
        loss = logistic_loss_fn(pred, y_train)

        # Backprop
        optimizer.zero_grad()
        model.zero_grad()
        with torch.no_grad():
            # Predictions on the training set
            pred = model(X_train)
            train_acc = accuracy_score(pred, y_train)

            # Predictions on the test set
            logits_test = model(X_test)
            test_acc = accuracy_score(logits_test, y_test)
            
            logits_clean = torch.matmul(X_train[clean_indices_train], model.p)  # Adjust to model.p shape
            probs_clean = F.softmax(logits_clean, dim=1)[:, 0].cpu().numpy()
            clean_probs_iter = np.mean(probs_clean)

            logits_noisy = torch.matmul(X_train[noisy_indices_train], model.p)  # Adjust to model.p shape
            probs_noisy = F.softmax(logits_noisy, dim=1)[:, 0].cpu().numpy()
            noisy_probs_iter = np.mean(probs_noisy)


        train_accs.append(train_acc)
        test_accs.append(test_acc)
        steps_list.append(cur_step)
        clean_proba.append(clean_probs_iter)
        noisy_proba.append(noisy_probs_iter)


        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            for param in model.parameters():
                param -= beta * param.grad
            
        # Logging tous les epochs
        cur_step += 1
        if (epoch==0) or (epoch+1)%log_every==0 or (epoch+1 == num_steps):
            print(f"Epoch {epoch+1}")


    plot_metrics(train_accs, test_accs, clean_proba, noisy_proba)
    return model


def train_with_max_margin(model, 
                        optimizer, 
                        X_train, 
                        y_train, 
                        X_test, 
                        y_test, 
                        clean_indices_train,
                        noisy_indices_train,
                        constraints,
                        log_every=50, 
                        num_steps=150,
                        beta=0.025, 
                        margin_lambda=1):

    # Initialize lists and step
    train_accs, test_accs, clean_proba, noisy_proba, steps_list = [0.01], [0.01], [0.5], [0.5], []
    cur_step = 0

    (r, R) = constraints

    # Update parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "p" in name:
                param -= beta * param.grad
                # Project p to satisfy norm constraint
                param_norm = torch.norm(param)
                if param_norm > R:
                    param.mul_(R / param_norm)
            elif "v" in name:
                param -= beta * param.grad
                # Project v to satisfy norm constraint
                param_norm = torch.norm(param)
                if param_norm > r:
                    param.mul_(r / param_norm)


    # Descent
    for epoch in range(num_steps):
        pred = model(X_train)
        margin_loss = -torch.min(y_train * pred)  # Maximize the margin by minimizing its negative
        regular_loss = logistic_loss_fn(pred, y_train)
        loss = regular_loss + margin_lambda * margin_loss

        # Backprop
        optimizer.zero_grad()
        model.zero_grad()
        with torch.no_grad():
            # Predictions on the training set
            pred = model(X_train)
            train_acc = accuracy_score(pred, y_train)

            # Predictions on the test set
            logits_test = model(X_test)
            test_acc = accuracy_score(logits_test, y_test)

            # Add accs to lists
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            logits_clean = torch.matmul(X_train[clean_indices_train], model.p)  # Adjust to model.p shape
            probs_clean = F.softmax(logits_clean, dim=1)[:, 0].cpu().numpy()
            clean_probs_iter = np.mean(probs_clean)

            logits_noisy = torch.matmul(X_train[noisy_indices_train], model.p)  # Adjust to model.p shape
            probs_noisy = F.softmax(logits_noisy, dim=1)[:, 0].cpu().numpy()
            noisy_probs_iter = np.mean(probs_noisy)


            # clean_probs_iter = np.mean(F.softmax(np.dot((X_train[clean_indices_train], model.p).cpu()))[:, 0])
            # noisy_probs_iter = np.mean(F.softmax(np.dot((X_train[noisy_indices_train], model.p).cpu()))[:, 0])


        steps_list.append(cur_step)
        clean_proba.append(clean_probs_iter)
        noisy_proba.append(noisy_probs_iter)


        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            for param in model.parameters():
                param -= beta * param.grad
            
        # Logging tous les epochs
        cur_step += 1
        if (epoch==0) or (epoch+1)%log_every==0 or (epoch+1 == num_steps):
            print(f"Epoch {epoch+1}")


    plot_metrics(train_accs, test_accs, clean_proba, noisy_proba)
    return model

