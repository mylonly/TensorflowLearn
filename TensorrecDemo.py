import numpy as np
import tensorrec

model = tensorrec.TensorRec()

interactions, user_features, item_features = tensorrec.util.generate_dummy_data(
    num_users=150,
    num_items=100,
    interaction_density=.05
)

model.fit(interactions, user_features, item_features, epochs=5, verbose=True)

predictions = model.predict(user_features=user_features,
                            item_features=item_features)


predicted_ranks = model.predict_rank(user_features=user_features,
                                     item_features=item_features)

r_at_k = tensorrec.eval.recall_at_k(predicted_ranks, interactions, k=10)

print(np.mean(r_at_k))