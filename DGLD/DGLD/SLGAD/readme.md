Process Step:

1. **Graph View Establishment for Anomaly Detection**
   
   1. **Target node sampling.** sample a target node from the given input graph via the **uniform sampling without replacement**.
   2. **Graph view sampling.** Alternatively, in this paper, we leverage **random walks with restart (RWR)** [44] as augmentations, which avoid violating the underlying graph semantic information. **Specififically, this approach samples graph views centered at a target node with the fifixed size *K***, which controls the radius of surrounding contexts. It is worth noting that graph diffusion [45] could also be a possible augmentation in our method to further injects theglobal information into our multi-view contrastiveness, which we leave in our future work.
   3. **Graph view anonymization.**
   
2. **Generative Learning with Attribute Reconstruction**
   
   1. ${x}' = AE(x) = f_{dec}(f_{enc}(x))$
   
   2. **GNN-based encoder**
   
      $\textbf{H}_{\phi_i} = GNN_{enc}(\textbf{X}_t^{\phi_i}, \textbf{A}_t^{\phi_i})$
   
      $\textbf{m}_{\phi_i}^{(l)} = AGGREGATE^{(l)}(\textbf{H}_{\phi_i}^{(l - 1)}[k, :]: v_k \in \mathcal{N}(v_j))$
   
      $\textbf{H}_{\phi_i}^{(l)} = COMBINE^{(l)}(\textbf{H}_{\phi_i}^{(l - 1)}[j, :]: \textbf{m}_{\phi_i}^{(l - 1)}[j, :])$
   
      $\textbf{H}_{\phi_i} = GNN_{enc}(\textbf{X}_t^{\phi_i}, \textbf{A}_t^{\phi_i}) = \sigma(\hat{\textbf{A}}^{\phi_i} \textbf{X}_t^{\phi_i} \textbf{W}_{enc})$
   
   3. **GNN-based decoder**
   
      $\textbf{X}_t^{\phi_i} = GNN_{dec}(\textbf{H}_{\phi_i}, \textbf{A}_t^{\phi_i}) = \sigma(\hat{\textbf{A}}^{\phi_i} \textbf{H}_{\phi_i} \textbf{W}_{dec})$
   
   4. **Generative graph anomaly detection**
   
      $\mathcal{L}_{gen}^j = \frac{1}{N} \sum_{i = 1}^N (\hat{\textbf{X}}_i^{\phi_j}[-1, :], \textbf{x}_i)^2$
   
3. **Multi-View Contrastive Learning**

   1. **GNN-based encoder**

      $\textbf{h}_t = \sigma(\textbf{x}_t \textbf{W}_{enc})$

      $\textbf{g}_{\phi_i} = \frac{1}{K} \sum_{i = 1}^K \textbf{H}_{\phi_i}[j, :]$

   2. **Contrastive module**

      $P_t^{\phi_i} = (\textbf{h}_t, \textbf{g}_{\phi_i})$

      $\widetilde{P}_t^{\phi_i} = (\textbf{h}_t, \widetilde{\textbf{g}}_{\phi_i})$

      $s_t^{\phi_i} = \sigma(\textbf{h}_t \textbf{W}_s \textbf{g}_{\phi_i}^T)$

      $\widetilde{s}_t^{\phi_i} = \sigma(\textbf{h}_t \textbf{W}_s \widetilde{\textbf{g}}_{\phi_i}^T)$

      $\mathcal{L}_{con}^j = - \frac{1}{2N} \sum_{i = 1}^N (log(s_i^{\phi_j}) + log(1 - \widetilde{s}_i^{\phi_j}))$

4. **Graph Anomaly Scoring**

   $f_{gen}(v_i) = \frac{1}{2} \sum_{j = 1}^2 (\delta^1||\hat{\textbf{X}}_i^{\phi_j}[-1, :] - \textbf{x}_i||_2^2)$

   $f_{con}(v_i) = \frac{1}{2} \sum_{j = 1}^2 \delta^2(\widetilde{s}_i^{\phi_j} - s_i^{\phi_j})$

   $f(v_i) = \alpha f_{con}(v_i) + \beta f_{gen}(v_i)$

   

   

   