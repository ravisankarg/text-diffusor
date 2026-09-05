# Multilingual content classification for on-device search

Status: design direction; not yet implemented or benchmarked in this repository.

## Executive summary

The masked-diffusion experiment proves that Granite Embedding 97M can be given a
masked-language-model head and trained as an iterative generator. That mechanism is
interesting, but generation is not the best fit for short, controlled search metadata:
the current decoder needs one complete encoder pass for every committed output token.

A more useful search direction is to retain Granite as an encoder and attach small
classification heads. One encoder pass over a paragraph can then produce both its
existing semantic embedding and three complementary, language-neutral signals:

1. **Domain** — what the content is about.
2. **Content kind** — what type of information object it is.
3. **Facets** — which searchable facts or affordances it contains.

These predictions should be created once during indexing, stored beside the ordinary
text and embedding index, and used as calibrated ranking features. They should not
replace lexical retrieval, dense retrieval, or authoritative metadata filters.

The product opportunity is an inspectable semantic layer shared by indexing, the Query
Planner (QP), search execution, ranking, and result explanations. For example, a query
such as `when does my car cover expire?` could find OCR text containing `motor policy
valid until ...` through compatible `insurance`, `vehicle`, and `date` signals even
when the literal wording differs.

## Why three axes instead of one label

A single category taxonomy creates brittle compound labels such as
`health_receipt`, `travel_receipt`, and `sports_receipt`. Separate heads reuse evidence
and generalize to combinations absent from training. A pharmacy receipt can be both a
receipt and health-related; a doctor-appointment notification can be both a reminder
and an event.

All predicted axes should be multi-label. Internally they should use stable IDs and
confidence values, not free-form generated strings. Friendly strings are presentation
labels only.

### Domain: subject matter

An initial controlled set could include:

- health
- finance
- transport
- travel
- sports
- education
- work
- government
- shopping
- entertainment
- technology
- home
- legal
- personal
- other

`transport` and `travel` need written definitions if both remain. One reasonable
boundary is that transport covers vehicles, commuting, and transit operations, while
travel covers trips, lodging, itineraries, and tourism. Labels that annotators cannot
distinguish consistently should be merged.

### Content kind: information-object type

An initial set could include:

- receipt
- invoice
- bill
- ticket
- booking
- reminder
- event
- memo
- note
- message
- email
- identity document
- certificate
- prescription
- report
- list
- form
- article
- code
- other

Some kinds legitimately co-occur. A train purchase may be both `ticket` and `receipt`;
an appointment message may be `message`, `reminder`, and `event`. `memo` and `note`
should remain separate only if they produce different search or presentation behavior.

### Facets: useful information present

An initial set could include:

- amount
- date
- deadline
- person
- location
- contact
- identifier
- status
- action

Facets do not claim that a value was extracted correctly. They say that a type of
information appears to be present and can guide retrieval or a later bounded extractor.

### Why facets improve search

A facet is the bridge between finding a relevant record and finding a record that can
answer the question. Domain identifies the subject, content kind identifies the
information object, and facets identify the answerable fields present in that object.

| Query | Domain | Content kind | Requested facet |
| --- | --- | --- | --- |
| `When is my dentist appointment?` | health | event / booking | date |
| `Where is the appointment?` | health | event / booking | location |
| `How much was the electricity bill?` | home / finance | bill | amount |
| `What is my train booking number?` | travel | booking | identifier |
| `Has my refund arrived?` | shopping / finance | receipt / message | status |

Two records may both be classified as `health` and `event`, while only one contains a
date. Facet compatibility lets ranking prefer the record that is not merely related but
is capable of answering the query.

The two sides should be named explicitly:

- `available_facets` are predicted once for an indexed record.
- `requested_facets` are produced from the user's query by QP.

For example:

```json
{
  "record": {"available_facets": {"date": 0.94, "location": 0.81}},
  "query_plan": {"requested_facets": ["date"]}
}
```

At ranking time, requested-facet coverage is a cheap soft feature over already
retrieved candidates. After ranking, the same request can route the winning evidence to
a bounded date, amount, location, or identifier extractor. It can also support a
grounded explanation such as `Matched appointment date`.

This separation is especially useful across languages: query expressions equivalent
to `when` can all map to the internal `date` ID even when the stored record uses a
different language or wording.

A facet is not the extracted value. An `available_facets.date` score of `0.94` means
that a date probably appears in the record; it does not identify the correct date or
prove what that date means. Extraction and evidence validation must still determine the
answer.

The first pilot should use only facets with an explicit search behavior: `date`,
`deadline`, `amount`, `location`, `contact`, `identifier`, and `status`. Broader facets
such as `person` or `action` should be added only when they produce a measurable ranking,
extraction, or explanation benefit. The facet head should remain optional until an
ablation demonstrates better Recall@K, ranking quality, or answer-evidence hit rate than
embeddings plus domain and content kind alone.

### Known metadata is not a prediction target

The system already knows whether content came from a note, file, message, or image OCR.
It should store that as authoritative `source_kind` metadata rather than asking the
classifier to infer it. File MIME type, timestamp, application ownership, and known
person IDs should remain authoritative as well.

Example indexed representation:

```json
{
  "source_kind": "image_ocr",
  "domains": {"health": 0.97, "finance": 0.58},
  "content_kinds": {"receipt": 0.91, "prescription": 0.63},
  "available_facets": {"amount": 0.96, "date": 0.89}
}
```

## Model architecture

The simplest design reuses one pooled Granite representation:

```text
text or OCR paragraph
        |
        v
Granite Embedding 97M encoder
        |
        +----> 384-dimensional retrieval embedding
        |
        +----> sigmoid(domain head)
        |
        +----> sigmoid(content-kind head)
        |
        +----> sigmoid(facet head)
```

Each head can be a small linear layer over the same 384-dimensional representation.
For dozens of labels, its cost is negligible relative to the encoder. If the product
already encodes the paragraph with another capable multilingual text encoder, attaching
the heads to that existing representation should be tested before loading a second
97M-parameter model solely for classification.

The first experiment should start from the original
`ibm-granite/granite-embedding-97m-multilingual-r2` checkpoint, not the locally trained
conversation-diffusion checkpoint. The latter was optimized for masked response
reconstruction, not calibrated classification, and its reported generative behavior is
not evidence for multilingual topic quality.

Training should begin with the encoder frozen. If a linear probe is insufficient,
unfreeze only upper encoder layers or use a parameter-efficient adapter. Any encoder
fine-tuning must retain a multilingual contrastive retrieval objective so that better
classification does not silently damage the embedding space.

## Index-time and query-time integration

### Index time

```text
source ingestion
    -> text extraction or OCR
    -> paragraph/chunk normalization
    -> one encoder pass
    -> embedding + domain/kind/facet probabilities
    -> persistent searchable record
```

Classification belongs in the background derived-index pipeline. It should run only
for new or changed chunks, preserve stable record identity, and be independently
versioned so a taxonomy or model upgrade can rebuild derived tags without rerunning OCR
or deleting the source index.

### Query time

Using the existing QP/STE/STR terminology, the intended flow is:

```text
QP
  -> hard structured constraints
  -> query-domain probabilities
  -> expected content-kind probabilities
  -> requested facets
        |
        v
STE / retrieval execution
  -> authoritative filters
  -> lexical and dense candidates
        |
        v
STR / fusion and ranking
  -> base relevance
  -> domain compatibility
  -> content-kind compatibility
  -> facet coverage
  -> ranked records and grounded match reasons
```

For `when is my next dentist appointment?`, a plan might contain:

```json
{
  "domains": ["health"],
  "expected_content_kinds": ["event", "reminder", "booking"],
  "requested_facets": ["date", "location"]
}
```

The ranking implementation can compare probability vectors rather than strings:

```text
domain compatibility = similarity(query domain scores, record domain scores)
kind compatibility  = similarity(query kind scores, record kind scores)

final rank = fusion(
    lexical rank,
    dense rank,
    structured matches,
    domain compatibility,
    kind compatibility,
    requested-facet coverage
)
```

No fixed weights are proposed before an offline relevance benchmark. Content kind will
often be more discriminative than domain, but that must be measured rather than assumed.

### Hard versus soft behavior

Person, explicit time ranges, MIME/source restrictions, and other authoritative fields
can remain hard constraints. Learned domain and content-kind predictions should begin as
soft boosts because one false label must not make the correct record unrecoverable.

Even when a query explicitly says `receipt`, a safe early implementation can search the
full authoritative scope, strongly boost receipt-like records, and retain a lower-ranked
fallback path. Promotion to a hard model-authored filter requires measured per-language
recall and calibrated abstention.

User-visible explanations can expose high-confidence grounded labels, for example:

```text
Health · Appointment
Matched date and location information
```

They should not show a label when confidence is weak or when it conflicts with the
record's authoritative metadata.

## Thirty-language data strategy

No single public dataset represents arbitrary text saved on phones. The training mix
must separately cover multilingual semantic transfer, query language, long documents,
document types, OCR corruption, short fragments, code-switching, and open-set content.

### Backbone-language boundary

IBM's current model card says the underlying encoder was pretrained on 200+ languages
and received enhanced retrieval/cross-lingual training for 52. That is enough to make a
30-language classifier plausible, but it is not a guarantee for an arbitrary language
list. The enhanced set includes Bengali, Hindi, Marathi, Telugu, and Urdu, but does not
list several other major Indic languages, including Tamil, Kannada, Malayalam,
Gujarati, Punjabi, Assamese, and Odia.

The exact target-language list must therefore be fixed before corpus creation. For each
language, measure tokenizer expansion, frozen-encoder linear-probe quality, OCR quality,
and cross-language query-to-record retrieval. Languages outside the enhanced set should
be treated as risk slices rather than hidden inside a global average.

### Public foundations

Useful public foundations include:

| Dataset | Useful contribution | Important limitation |
| --- | --- | --- |
| MASSIVE | More than one million parallel virtual-assistant utterances across 51 languages, 18 domains, 60 intents, and 55 slots; useful for query-side intent and translation consistency | Queries rather than stored documents; its intent schema is not the product taxonomy |
| SIB-200 | Topic classification across 205 languages and dialects; useful for language coverage and per-language evaluation | Only seven broad news topics and roughly 701 training examples per language |
| MultiEURLEX | 65,000 aligned, multi-label legal documents in 23 languages; useful for long-text and cross-lingual multi-label mechanics | Strong legal and European-domain bias |
| WIKI-DOC / MULTIEURLEX-DOC | Multilingual rendered-document data; useful for document/OCR robustness experiments | Does not represent the full distribution of phone notes, messages, screenshots, and local forms |
| RVL-CDIP | 400,000 English document images across 16 kinds such as invoice, email, memo, form, and report | English only; source and redistribution terms require a separate license audit |

These should be auxiliary data or evaluation sets. They should not be mechanically
collapsed into the production taxonomy when their labels do not mean the same thing.
For example, MASSIVE can train an auxiliary query-intent head while task-specific query
examples train `expected_content_kinds`.

Licenses vary by dataset and version. Every acquired row should retain source,
version/hash, license, transformation history, and required attribution. In particular,
CC-BY and CC-BY-SA material should not be mixed into an untracked aggregate. Licensing
requirements must be reviewed before distributing a dataset or trained model.

### Task-specific PhoneText-30 corpus

The primary corpus should model what actually appears on phones. A useful record schema
is:

```text
record_id
semantic_group_id
language
script
source_kind
clean_text
observed_text
domains[]
content_kinds[]
facets[]
provenance
license
transformation_history
```

`semantic_group_id` connects an original example, translations, localized versions,
renderings, OCR outputs, and other augmentations. Train/validation/test splitting must
happen by this group before translation or augmentation. Otherwise, the same semantic
example can leak across splits in different languages and make cross-lingual scores look
far better than real generalization.

The corpus should include:

1. **Task-specific clean content.** Notes, reminders, receipts, bills, invoices,
   tickets, bookings, forms, messages, saved articles, lists, reports, code fragments,
   and general prose.
2. **Native localization.** Local currencies, dates, addresses, transport systems,
   government-document names, abbreviations, and natural phrasing. Literal translation
   alone is insufficient.
3. **Cross-language retrieval pairs.** Queries and relevant records should sometimes be
   in different languages, such as an English query over a Telugu record.
4. **Code-switching and transliteration.** Mixed scripts, Romanized regional languages,
   English product names, emojis, URLs, filenames, and incomplete fragments.
5. **Open-set negatives.** Poetry, recipes, logs, copied webpages, game text, random
   snippets, source code, and other material that should remain `other` or receive only
   broad labels.

### OCR data should match the deployed engine

Generic character corruption is helpful but insufficient. The preferred augmentation
loop is:

```text
clean labeled text
    -> render with target-language fonts and realistic layouts
    -> vary resolution, blur, contrast, compression, skew, crop, and background
    -> run the same OCR engine and configuration used on the phone
    -> retain both clean_text and observed_text with identical semantic labels
```

This captures script- and engine-specific mistakes, broken word boundaries, table/line
reordering, lost diacritics, and partial extraction. Training should include clean-only,
OCR-only, and clean/OCR consistency objectives. Synthetic names and identifiers should
be used; private user content should remain on-device unless a user explicitly opts in
to a separately governed contribution flow.

### Suggested scale and rollout

A practical first full-corpus target is 25-40 useful labels, 200-500 localized examples
per label and target language, plus at least one noisy or OCR view. Because examples can
carry several labels, the exact multiplication is not literal, but approximately
300,000-800,000 balanced rows is a reasonable experiment scale rather than a quality
guarantee.

Before producing the full set, run a smaller pilot on approximately six languages chosen
to cover distinct scripts, resource levels, and OCR behavior. A successful pilot must
improve retrieval, not merely classification accuracy. Then expand to all 30 languages
with language-balanced and label-balanced sampling.

A separately curated, human-reviewed evaluation set should contain at least 150-300
examples per target language. Machine-translated labels alone cannot validate native
wording, culturally specific records, code-switching, or real OCR behavior.

## Training plan

1. Freeze the original Granite encoder and train the three linear heads as a baseline.
2. Use independent sigmoid/BCE objectives with class balancing and per-label thresholds.
3. Train clean and OCR/noisy views to agree while preserving their gold multi-label
   targets.
4. Balance batches across languages, scripts, source kinds, and labels so English and
   common classes do not dominate.
5. If the frozen probe is insufficient, unfreeze only upper layers or add an adapter.
6. Preserve embedding quality with multilingual query-record contrastive pairs and hard
   negatives during any encoder update.
7. Calibrate thresholds on held-out native data, including an explicit abstain/other
   policy.
8. Optionally distill category probabilities from a larger multilingual teacher using
   public or synthetic content. Do not upload private phone text to a teacher service.

The classifier can later support local personalization from explicit corrections, but
the base release must work without collecting personal content.

## Evaluation and acceptance gates

### Classification quality

Report at minimum:

- macro and micro F1 for each head
- per-language and per-label F1
- worst-language results alongside averages
- multi-label precision and recall
- calibration error and threshold stability
- other/abstention precision and recall
- clean-to-OCR degradation by OCR error bucket
- code-mixed and transliterated slices
- short fragments, long paragraphs, and mixed-domain records

### Search quality

Classifier accuracy alone is not the product success criterion. Compare the existing
pipeline against the classifier-augmented pipeline on labeled queries using:

- Recall@K
- MRR and/or nDCG
- zero-result rate
- false-negative rate introduced by category logic
- cross-language query-to-record retrieval
- gains by language, source kind, domain, and content kind
- quality of user-visible match reasons

Required ablations should include base hybrid retrieval, domain only, content kind
only, facets only, all classifier signals, and any proposed hard-filter variant.

### Device quality

Measure separately:

- incremental indexing latency per chunk
- throughput by input length
- peak memory
- model and head size
- battery and thermal behavior during background indexing
- query-time latency added by query classification
- rebuild time after a taxonomy or model-version change

Desktop accuracy and a successful model conversion do not establish acceptable Android
latency or battery behavior.

## Non-goals and safeguards

- This direction does not replace lexical search, dense embeddings, or structured
  metadata.
- It does not treat generated labels as factual extracted values.
- It does not infer source facts that the platform already provides.
- It does not hard-filter records from uncalibrated model predictions.
- It does not claim thirty-language quality from a multilingual model card alone.
- It does not use the current diffusion checkpoint's generation results as classifier
  evidence.
- It does not require uploading private user content.

## Decisions required before implementation

1. Exact 30 target languages and required code-mixed/transliterated variants.
2. Written label definitions, merge rules, and positive/negative examples.
3. Whether domain, content-kind, and facet labels alter retrieval, result explanation,
   evidence selection, or all three.
4. Which existing text encoder can expose the shared representation on the target
   Android runtime.
5. The dataset license policy for redistributed data and trained artifacts.
6. Offline relevance benchmark, annotation protocol, and minimum acceptance thresholds.
7. Storage schema and versioned background rebuild behavior.

## References

- IBM Granite Embedding 97M Multilingual R2 model card:
  https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2
- MASSIVE dataset:
  https://www.amazon.science/code-and-datasets/massive
- SIB-200 dataset:
  https://huggingface.co/datasets/Davlan/sib200
- MultiEURLEX paper and dataset description:
  https://aclanthology.org/2021.emnlp-main.559/
- WIKI-DOC and MULTIEURLEX-DOC dataset:
  https://huggingface.co/datasets/AmazonScience/MultilingualMultiModalClassification
- RVL-CDIP dataset:
  https://adamharley.com/rvl-cdip/
- SPLADE v2, an example of precomputed masked-LM-derived sparse retrieval signals:
  https://arxiv.org/abs/2109.10086
- Federated multi-domain RAG with query-domain routing and unified relevance scoring:
  https://aclanthology.org/2025.coling-industry.33/

## Current conclusion

The strongest form of this idea is not a two-word diffusion generator. It is a shared,
one-pass multilingual encoder that emits a retrieval embedding plus calibrated domain,
content-kind, and facet distributions. Those distributions become versioned index
metadata, soft relevance features, and grounded user explanations. The design is
promising, but its value must be established by multilingual, OCR-aware retrieval
benchmarks before it becomes a hard search dependency.
