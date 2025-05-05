import tensorflow as tf


# Moe with hard gatting
class MoEModel(tf.keras.Model):
    def __init__(self, gate, experts):
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.num_experts = len(experts)

    def call(self, x, training=False):
        gate_logits = self.gate(x, training=training)
        gate_probs = tf.nn.softmax(gate_logits, axis=-1)
        expert_idx = tf.argmax(gate_probs, axis=-1)

        expert_outputs = [expert(x, training=training)
                          for expert in self.experts]
        stacked = tf.stack(expert_outputs, axis=1)
        mask = tf.one_hot(expert_idx, depth=self.num_experts)
        mask = tf.expand_dims(mask, -1)

        output = tf.reduce_sum(stacked * mask, axis=1)

        self.last_expert_idx = int(expert_idx.numpy()[0])

        return output
