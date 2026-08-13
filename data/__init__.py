"""
Data generation, decoupled from training.

    from data.init_data import generate
    train_data, val_data = generate(operator="+", modulus=97, train_pct=50)

See data.init_data.generate for the full parameter list. The resulting
(train_data, val_data) tuple is a pair of grok.data.ArithmeticDataset, ready
to pass straight into nn.relu.train(train_data=..., val_data=...) or
nn.gelu.train(train_data=..., val_data=...).
"""
