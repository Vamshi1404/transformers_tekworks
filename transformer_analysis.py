import tensorflow as tf

from tensorflow.keras.layers import (
    TextVectorization,
    Embedding,
    Dense,
    LayerNormalization,
    MultiHeadAttention
)

from tensorflow.keras import Model

data = [
    ("i am a student", "nenu vidyarthini"),
    ("how are you", "meeru ela unnaru"),
    ("i love machine learning", "nenu machine learning ishtapadatanu"),
    ("good morning", "subhodayam"),
    ("thank you", "dhanyavadalu"),
    ("see you later", "tarvata kalustanu"),
    ("what is your name", "mee peru emi"),
    ("where are you going", "meeru ekkadi veltunnaru"),
    ("i like coffee", "nenu coffee ishtapadatanu"),
    ("welcome", "swagatam"),
    ("good night", "subharatri"),
    ("i am happy", "nenu santoshamga unnanu"),
    ("where do you live", "meeru ekkada untaru"),
    ("i am learning ai", "nenu ai nerchukuntunnanu"),
    ("how is your day", "mee roju ela undi"),
    ("nice to meet you", "mimmlani kalavadam santosham"),
    ("have a nice day", "mee roju bagundali"),
    ("i love programming", "naku programming ante ishtam"),
    ("what are you doing", "meeru emi chestunnaru"),
    ("see you tomorrow", "repu kaluddam")
]

english_sentences = [
    x[0]
    for x in data
]

telugu_sentences = [
    "start " + x[1] + " end"
    for x in data
]

vocab_size = 2000

sequence_length = 12

source_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length
)

target_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length
)

source_vectorization.adapt(
    english_sentences
)

target_vectorization.adapt(
    telugu_sentences
)

encoder_inputs_data = source_vectorization(
    english_sentences
)

target_tokens = target_vectorization(
    telugu_sentences
)

decoder_inputs_data = target_tokens[:, :-1]

decoder_targets = target_tokens[:, 1:]


class PositionalEmbedding(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        sequence_length,
        vocab_size,
        embed_dim
    ):

        super().__init__()

        self.token_embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )

        self.position_embedding = Embedding(
            input_dim=sequence_length,
            output_dim=embed_dim
        )

    def call(
        self,
        inputs
    ):

        length = tf.shape(
            inputs
        )[-1]

        positions = tf.range(
            start=0,
            limit=length,
            delta=1
        )

        token_embeddings = self.token_embedding(
            inputs
        )

        position_embeddings = self.position_embedding(
            positions
        )

        return (
            token_embeddings +
            position_embeddings
        )


class TransformerEncoder(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        embed_dim,
        dense_dim,
        num_heads
    ):

        super().__init__()

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.dense_proj = tf.keras.Sequential([
            Dense(
                dense_dim,
                activation="relu"
            ),
            Dense(
                embed_dim
            )
        ])

        self.layernorm1 = LayerNormalization()

        self.layernorm2 = LayerNormalization()

    def call(
        self,
        inputs
    ):

        attention_output = self.attention(
            query=inputs,
            value=inputs,
            key=inputs
        )

        proj_input = self.layernorm1(
            inputs + attention_output
        )

        proj_output = self.dense_proj(
            proj_input
        )

        return self.layernorm2(
            proj_input + proj_output
        )


class TransformerDecoder(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        embed_dim,
        dense_dim,
        num_heads
    ):

        super().__init__()

        self.self_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.cross_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn = tf.keras.Sequential([
            Dense(
                dense_dim,
                activation="relu"
            ),
            Dense(
                embed_dim
            )
        ])

        self.layernorm1 = LayerNormalization()

        self.layernorm2 = LayerNormalization()

        self.layernorm3 = LayerNormalization()

    def call(
        self,
        inputs,
        encoder_outputs
    ):

        attention_output = self.self_attention(
            query=inputs,
            value=inputs,
            key=inputs,
            use_causal_mask=True
        )

        out1 = self.layernorm1(
            inputs + attention_output
        )

        attention_output2 = self.cross_attention(
            query=out1,
            value=encoder_outputs,
            key=encoder_outputs
        )

        out2 = self.layernorm2(
            out1 + attention_output2
        )

        ffn_output = self.ffn(
            out2
        )

        return self.layernorm3(
            out2 + ffn_output
        )


def build_model():

    embed_dim = 128

    dense_dim = 256

    num_heads = 4

    encoder_input = tf.keras.Input(
        shape=(None,),
        dtype="int64"
    )

    decoder_input = tf.keras.Input(
        shape=(None,),
        dtype="int64"
    )

    encoder_embeddings = PositionalEmbedding(
        sequence_length,
        vocab_size,
        embed_dim
    )(encoder_input)

    encoder_output = TransformerEncoder(
        embed_dim,
        dense_dim,
        num_heads
    )(encoder_embeddings)

    decoder_embeddings = PositionalEmbedding(
        sequence_length,
        vocab_size,
        embed_dim
    )(decoder_input)

    decoder_output = TransformerDecoder(
        embed_dim,
        dense_dim,
        num_heads
    )(
        decoder_embeddings,
        encoder_output
    )

    final_output = Dense(
        vocab_size,
        activation="softmax"
    )(decoder_output)

    model = Model(
        [encoder_input, decoder_input],
        final_output
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def load_model():

    X = encoder_inputs_data

    y_in = decoder_inputs_data

    y_out = decoder_targets

    model = build_model()

    model.fit(
        [X, y_in],
        y_out,
        epochs=50,
        batch_size=2,
        verbose=0
    )

    return model


def translate(
    input_sentence,
    model
):

    tokenized_src = source_vectorization(
        [input_sentence.lower()]
    )

    decoded_tokens = ["start"]

    target_vocab = target_vectorization.get_vocabulary()

    target_index_lookup = dict(
        zip(
            range(
                len(target_vocab)
            ),
            target_vocab
        )
    )

    for _ in range(
        sequence_length
    ):

        tokenized_tgt = target_vectorization([
            " ".join(
                decoded_tokens
            )
        ])

        predictions = model(
            [
                tokenized_src,
                tokenized_tgt
            ],
            training=False
        )

        next_token_idx = tf.argmax(
            predictions[
                0,
                len(decoded_tokens) - 1,
                :
            ]
        ).numpy()

        next_token = target_index_lookup.get(
            next_token_idx,
            ""
        )

        if (
            next_token == "end"
            or
            next_token == ""
        ):

            break

        decoded_tokens.append(
            next_token
        )

    return " ".join(
        decoded_tokens[1:]
    )