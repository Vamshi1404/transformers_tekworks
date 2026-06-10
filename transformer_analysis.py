import os
import tensorflow as tf
from tensorflow.keras.layers import (
    TextVectorization,
    Embedding,
    Dense,
    LayerNormalization,
    MultiHeadAttention,
    Dropout,
)
from tensorflow.keras import Model
import numpy as np
import pickle

# ---------------------------------------------------------------------------
# Dataset  (English → Telugu)
# ---------------------------------------------------------------------------

RAW_DATA = [
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
    ("see you tomorrow", "repu kaluddam"),
    ("i am fine", "nenu bagunnanu"),
    ("what time is it", "ippudu samayam entandi"),
    ("i am tired", "nenuअllasipoyanu"),
    ("please help me", "dayachesi naaku saayam cheyandi"),
    ("i do not understand", "nenu artham chesukoledu"),
    ("can you speak slowly", "meeru nilldgaa maatladagalaraa"),
    ("i am from india", "nenu india nunchi vastanu"),
    ("it is very hot today", "ee roju chaala vedi ga undi"),
    ("i want to eat", "nenu tinavalani undi"),
    ("where is the hospital", "hospital ekkadi undi"),
    ("i need water", "naaku neellu kaavali"),
    ("how much does it cost", "idi entha aavutundi"),
    ("good afternoon", "shubha madhyanham"),
    ("i am busy", "nenu busy ga unnanu"),
    ("let us go", "velladaam"),
    ("come here", "ikkade randi"),
    ("i miss you", "nenu mee gurinchi miss avutunnanu"),
    ("be careful", "jaagratthaga undi"),
    ("all the best", "anni melde"),
    ("i am sorry", "nenu varintiki noppinchanu"),
    ("please sit down", "dayachesi koorchonddi"),
    ("open the door", "thaluppu theyandi"),
    ("close the window", "kiddiki meyyandi"),
    ("turn off the light", "veluturu aapandi"),
    ("i love reading books", "nenu books chaduvadam ishtapadatanu"),
    ("today is a holiday", "ee roju saati roju"),
    ("i will call you", "nenu mee ki phone chestanu"),
    ("take care of yourself", "mee gurinchi mee chusukonddi"),
    ("happy birthday", "janma dinotsava shubhakankshalu"),
    ("congratulations", "abhinavandanalu"),
    ("i am hungry", "naku aakali ga undi"),
    ("the food is delicious", "anna chaala ruchiga undi"),
    ("i want to sleep", "nenu nidravaalatani undi"),
    ("good luck", "mangalavaram meedha aasha"),
    ("i am cold", "nenu challaagaa unnanu"),
    ("turn on the fan", "fan veyandi"),
    ("i need help", "naaku saayam kaavali"),
    ("do you have time", "mee daggara time undaa"),
    ("i will be right back", "nenu ippude tirigivastaanu"),
    ("what happened", "emi jarigindi"),
    ("do not worry", "parishthampadakandi"),
    ("everything is fine", "anni bagunnaayi"),
    ("i love my family", "nenu naa kutumbam ni ishtapadatanu"),
    ("she is my sister", "aame naa chelikellu"),
    ("he is my brother", "atanu naa annayi"),
    ("my mother is kind", "naa amma dayagaladi"),
    ("my father is strong", "naa nanna balavanthudu"),
    ("i have a dog", "naake okka kukka undi"),
    ("the sky is blue", "aakasham neelanga undi"),
    ("it is raining outside", "baahata vana padutundi"),
    ("i like music", "nenu sangeetam ishtapadatanu"),
    ("let me know if you need anything", "meeru emaina kaavaalante cheppandi"),
    ("i am going home", "nenu intiki veltunnanu"),
    ("the train is late", "train aalachiga undi"),
    ("i lost my phone", "naa phone poyindi"),
    ("can you help me", "meeru naaku saayam cheyagalaara"),
    ("i do not know", "naaku teliyadhu"),
    ("this is very beautiful", "idi chaala andamga undi"),
    ("i want to learn telugu", "nenu telugu nerchukovalani undi"),
    ("speak louder please", "dayachesi gudduga maatladandi"),
    ("write it down", "adhi raayandi"),
    ("i finished my work", "nenu naa pani chesaanu"),
    ("well done", "baagaa chesaaru"),
    ("try again", "malli prayatninchaandi"),
    ("i am excited", "nenu chanipoyaanu chacha"),
    ("please wait", "dayachesi aagundi"),
    ("you are right", "meeru correct ga cheppindhi"),
    ("that is wrong", "adhi tappu"),
    ("i agree with you", "nenu mee tho agree avutanu"),
    ("it is getting late", "aaalachipotundi"),
    ("let us meet tomorrow", "repu kalustaam"),
    ("i am on my way", "nenu daaarin lo unnanu"),
    ("good job", "baaga chesaaru"),
    ("i am proud of you", "nenu mee gurinchi garvapadatanu"),
    ("i will try my best", "nenu naa chethalonaina chesataanu"),
    ("it was nice talking to you", "meeru tho maatladadam chaala santosham"),
    ("take your time", "mee samayam teesukonddi"),
    ("i need a break", "naaku vishranti kaavali"),
    ("i am almost done", "nenu daadaapu aipoyaanu"),
    ("that sounds good", "adhi baagaa undi"),
]

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

VOCAB_SIZE      = 3000
SEQUENCE_LENGTH = 14
EMBED_DIM       = 256
DENSE_DIM       = 512
NUM_HEADS       = 8
DROPOUT_RATE    = 0.1
EPOCHS          = 300
BATCH_SIZE      = 8

MODEL_PATH      = "saved_model/en_te_transformer.keras"
VECTORIZER_PATH = "saved_model/vectorizers.pkl"

# ---------------------------------------------------------------------------
# Prepare text data
# ---------------------------------------------------------------------------

english_sentences = [x[0] for x in RAW_DATA]
telugu_sentences  = ["start " + x[1] + " end" for x in RAW_DATA]

source_vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
)

target_vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
)

source_vectorization.adapt(english_sentences)
target_vectorization.adapt(telugu_sentences)

encoder_inputs_data  = source_vectorization(english_sentences)
target_tokens        = target_vectorization(telugu_sentences)
decoder_inputs_data  = target_tokens[:, :-1]
decoder_targets      = target_tokens[:, 1:]

# ---------------------------------------------------------------------------
# Model layers
# ---------------------------------------------------------------------------

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embedding    = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embedding = Embedding(input_dim=sequence_length, output_dim=embed_dim)

    def call(self, inputs):
        length    = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        return self.token_embedding(inputs) + self.position_embedding(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "sequence_length": self.position_embedding.input_dim,
            "vocab_size":      self.token_embedding.input_dim,
            "embed_dim":       self.token_embedding.output_dim,
        })
        return cfg


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention   = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj  = tf.keras.Sequential([
            Dense(dense_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1  = LayerNormalization()
        self.layernorm2  = LayerNormalization()
        self.dropout1    = Dropout(dropout_rate)
        self.dropout2    = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_out  = self.attention(query=inputs, value=inputs, key=inputs)
        attn_out  = self.dropout1(attn_out, training=training)
        proj_in   = self.layernorm1(inputs + attn_out)
        proj_out  = self.dense_proj(proj_in)
        proj_out  = self.dropout2(proj_out, training=training)
        return self.layernorm2(proj_in + proj_out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "embed_dim":    self.attention.key_dim,
            "dense_dim":    self.dense_proj.layers[0].units,
            "num_heads":    self.attention.num_heads,
            "dropout_rate": self.dropout1.rate,
        })
        return cfg


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.self_attention  = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.cross_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn             = tf.keras.Sequential([
            Dense(dense_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()
        self.dropout1   = Dropout(dropout_rate)
        self.dropout2   = Dropout(dropout_rate)
        self.dropout3   = Dropout(dropout_rate)

    def call(self, inputs, encoder_outputs, training=False):
        self_attn   = self.self_attention(
            query=inputs, value=inputs, key=inputs, use_causal_mask=True
        )
        self_attn   = self.dropout1(self_attn, training=training)
        out1        = self.layernorm1(inputs + self_attn)

        cross_attn  = self.cross_attention(
            query=out1, value=encoder_outputs, key=encoder_outputs
        )
        cross_attn  = self.dropout2(cross_attn, training=training)
        out2        = self.layernorm2(out1 + cross_attn)

        ffn_out     = self.ffn(out2)
        ffn_out     = self.dropout3(ffn_out, training=training)
        return self.layernorm3(out2 + ffn_out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "embed_dim":    self.self_attention.key_dim,
            "dense_dim":    self.ffn.layers[0].units,
            "num_heads":    self.self_attention.num_heads,
            "dropout_rate": self.dropout1.rate,
        })
        return cfg


# ---------------------------------------------------------------------------
# Build / train / save
# ---------------------------------------------------------------------------

def build_model():
    encoder_input = tf.keras.Input(shape=(None,), dtype="int64", name="encoder_input")
    decoder_input = tf.keras.Input(shape=(None,), dtype="int64", name="decoder_input")

    enc_emb = PositionalEmbedding(SEQUENCE_LENGTH, VOCAB_SIZE, EMBED_DIM)(encoder_input)
    enc_out = TransformerEncoder(EMBED_DIM, DENSE_DIM, NUM_HEADS, DROPOUT_RATE)(enc_emb)

    dec_emb = PositionalEmbedding(SEQUENCE_LENGTH, VOCAB_SIZE, EMBED_DIM)(decoder_input)
    dec_out = TransformerDecoder(EMBED_DIM, DENSE_DIM, NUM_HEADS, DROPOUT_RATE)(dec_emb, enc_out)

    final = Dense(VOCAB_SIZE, activation="softmax")(dec_out)

    model = Model([encoder_input, decoder_input], final)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_save():
    """Train the model on the full dataset and persist weights + vectorizers."""
    os.makedirs("saved_model", exist_ok=True)

    # Save vectorizers
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(
            {
                "source_config": source_vectorization.get_config(),
                "source_vocab":  source_vectorization.get_vocabulary(),
                "target_config": target_vectorization.get_config(),
                "target_vocab":  target_vectorization.get_vocabulary(),
            },
            f,
        )

    model = build_model()
    model.summary()

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=20, min_lr=1e-6, verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=40, restore_best_weights=True
    )

    history = model.fit(
        [encoder_inputs_data, decoder_inputs_data],
        decoder_targets,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[lr_scheduler, early_stop],
        verbose=1,
    )

    model.save(MODEL_PATH)
    print(f"\n✅  Model saved to {MODEL_PATH}")
    return model, history


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

_CUSTOM_OBJECTS = {
    "PositionalEmbedding": PositionalEmbedding,
    "TransformerEncoder":  TransformerEncoder,
    "TransformerDecoder":  TransformerDecoder,
}


def load_saved_model():
    """Load persisted model + rebuild vectorizers from saved vocab."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No saved model found at '{MODEL_PATH}'. "
            "Run `python transformer_analysis.py` first to train."
        )

    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=_CUSTOM_OBJECTS)

    with open(VECTORIZER_PATH, "rb") as f:
        vdata = pickle.load(f)

    src_vec = TextVectorization.from_config(vdata["source_config"])
    src_vec.set_vocabulary(vdata["source_vocab"])

    tgt_vec = TextVectorization.from_config(vdata["target_config"])
    tgt_vec.set_vocabulary(vdata["target_vocab"])

    return model, src_vec, tgt_vec


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def translate(input_sentence: str, model, src_vec, tgt_vec) -> str:
    tokenized_src = src_vec([input_sentence.lower()])
    decoded_tokens = ["start"]

    target_vocab       = tgt_vec.get_vocabulary()
    target_index_lookup = dict(enumerate(target_vocab))

    for _ in range(SEQUENCE_LENGTH):
        tokenized_tgt = tgt_vec([" ".join(decoded_tokens)])
        predictions   = model(
            [tokenized_src, tokenized_tgt], training=False
        )
        next_idx   = tf.argmax(
            predictions[0, len(decoded_tokens) - 1, :]
        ).numpy()
        next_token = target_index_lookup.get(next_idx, "")

        if next_token in ("end", ""):
            break
        decoded_tokens.append(next_token)

    return " ".join(decoded_tokens[1:])


def get_training_pairs():
    """Return all (english, telugu) pairs for display in the UI."""
    return [(e, t.replace("start ", "").replace(" end", "")) for e, t in
            zip(english_sentences, [x[1] for x in RAW_DATA])]


# ---------------------------------------------------------------------------
# Entry-point: train & smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model, history = train_and_save()

    # quick smoke test
    tests = [
        "good morning",
        "i am happy",
        "thank you",
        "where are you going",
        "i love programming",
    ]
    model_loaded, sv, tv = load_saved_model()
    print("\n--- Smoke-test translations ---")
    for t in tests:
        print(f"  {t!r:35s} → {translate(t, model_loaded, sv, tv)!r}")