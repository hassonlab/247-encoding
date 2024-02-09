# Non-configurable paramters. Don't touch.
FILE := tfsenc_main
USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d-%H%M")
DT := ${USR}

# -----------------------------------------------------------------------------
#  Configurable options
# -----------------------------------------------------------------------------

PRJCT_ID := tfs
# {podcast | tfs}

############## tfs electrode ids ##############
# 625 Electrode IDs
SID := 625
E_LIST := $(shell seq 1 105)
BC := 

# 676 Electrode IDs
# SID := 676
# E_LIST := $(shell seq 1 125)
# BC := --bad-convos 38 39

# 717 Electrode IDs
# SID := 7170
# E_LIST := $(shell seq 1 256)
# BC :=

# 798 Electrode IDs
# SID := 798
# E_LIST := $(shell seq 1 198)
# BC :=

# Sig file will override whatever electrodes you choose
SIG_FN := 
# SIG_FN := --sig-elec-file test.csv
# SIG_FN := --sig-elec-file 129-phase-5000-sig-elec-glove50d-perElec-FDR-01-LH.csv
# SIG_FN := --sig-elec-file colton625.csv colton625.csv
# SIG_FN := --sig-elec-file tfs-sig-file-625-top-0.3-prod.csv tfs-sig-file-625-sig-0.3-comp.csv
# SIG_FN := --sig-elec-file 625-mariano-prod-new-53.csv 625-mariano-comp-new-30.csv # for sig-test
# SIG_FN := --sig-elec-file 676-mariano-prod-new-109.csv 676-mariano-comp-new-104.csv # for sig-test
# SIG_FN := --sig-elec-file 7170-comp-sig.csv 7170-prod-sig.csv
# SIG_FN := --sig-elec-file tfs-sig-file-625-sig-1.0-comp.csv tfs-sig-file-625-sig-1.0-prod.csv
# SIG_FN := --sig-elec-file tfs-sig-file-7170-region-ifg.csv tfs-sig-file-7170-region-ifg.csv

# podcast electrode IDs
# SID := 777
# SID := 661
# E_LIST :=  $(shell seq 1 115)
# SID := 662
# E_LIST :=  $(shell seq 1 100)
# SID := 717
# E_LIST :=  $(shell seq 1 255)
# SID := 723
# E_LIST :=  $(shell seq 1 165)
# SID := 741
# E_LIST :=  $(shell seq 1 130)
# SID := 742
# E_LIST :=  $(shell seq 1 175)
# SID := 743
# E_LIST :=  $(shell seq 1 125)
# SID := 763
# E_LIST :=  $(shell seq 1 80)
# SID := 798
# E_LIST :=  $(shell seq 1 195)
#

### podcast significant electrode list (if provided, override electrode IDs)
# SIG_FN := --sig-elec-file podcast_160.csv

PKL_IDENTIFIER := full
# {full | trimmed}

# number of permutations (goes with SH and PSH)
NPERM := 1000

# Choose the lags to run for.
LAGS := {400000..500000..100} # lag400500k-100
LAGS := {-150000..150000..100} # lag60k-1k
LAGS := {-500..500..5} # lag500-5
LAGS := -300000 -250000 -200000 200000 250000 300000 # lag300k-50k
LAGS := -150000 -120000 -90000 90000 120000 150000 # lag150k-30k
LAGS := -60000 -50000 -40000 -30000 -20000 20000 30000 40000 50000 60000 # lag60k-10k
LAGS := {-10000..10000..25} # lag10k-25
LAGS := {-2000..2000..25} # lag2k-25
LAGS := {-5000..5000..25} # lag5k-25

# Conversation ID (Choose 0 to run for all conversations)
CONVERSATION_IDX := 0

# Choose which set of embeddings to use
# {glove50 | gpt2-xl | blenderbot-small}
EMB := blenderbot
EMB := gpt2-xl
EMB := blenderbot-small
EMB := gpt2-xl
CNXT_LEN := 1024

# Choose the window size to average for each point
WS := 200

# Choose which set of embeddings to align with (intersection of embeddings)
ALIGN_WITH := blenderbot-small
ALIGN_WITH := gpt2-xl
ALIGN_WITH := glove50 gpt2-xl blenderbot-small
ALIGN_WITH := $(EMB)
ALIGN_WITH := 

# Choose layer of embeddings to use
# {1 for glove, 48 for gpt2, 8 for blenderbot encoder, 16 for blenderbot decoder}
LAYER_IDX := 48

# Choose whether to PCA (0 or for no pca)
PCA_TO := 50

# Specify the minimum word frequency (0 for 247, 5 for podcast)
MWF := 0

# Specify the number of folds (usually 10)
FN := 10

# TODO: explain this parameter.
WV := all

# Choose whether to label or phase shuffle
# SH := --shuffle
# PSH := --phase-shuffle

# Choose whether to normalize the embeddings
NM := l2
# {l1 | l2 | max}

############## Embedding Modification ##############
# {-rand: random datum (random embeddings)}
# {-arb: arbitrary datum (arbitrary embeddings, same for same word)}

# {shift-emb: shifts embeddings (eg, from n-1 to n)}
# {shift-emb1: shifts embeddings (eg, from n-1 to n)}
# {shift-emb2: shifts embeddings 2 times (eg, from n-1 to n+1)}
# ... etc
# {shift-embn: shifts embeddings (eg, from n-1 to n-2)}
# {shift-embn1: shifts embeddings (eg, from n-1 to n-2)}
# {shift-embn2: shifts embeddings 2 times (eg, from n-1 to n-3)}
# ... etc

# {concat-emb: concat embeddings (eg, n-1 + n)}
# {concat-emb2: concat embeddings (eg, n-1 + n + n+1)}
# ... etc

# {glove50: force glove embeddings for glove50 pred}

EM := shift-emb
EM := 
EM := glove50


############## Datum Modifications ##############
# Add specific tags concatenated by '-'. The available tags are as below:

# 1. {no-trim}
#	if 'no-trim' is a substring of DM, do not trim datum words that have any lag \
outside of the conversation range (currently not used)
#	if 'no-trim' is not a substring of DM, datum will be trimmed based on maximum lag

# 2. Token manipulation {-all, -zeroshot, -correctn, -incorrectn, -probx, -improbx, -pred}
#	for all emb_type:
#	{all: choose all words}
#   {zeroshot: zeroshot datum (unique set of words)}

#	for emb_type other than glove:
#	{correctn: choose words correctly predicted by the model (with rank <= n)}
#	{incorrectn: choose words incorrectly predicted by the model (with rank > n)}
#	{probx: choose words with high predictability (with true_pre_prob >= top x-th percentile)}
#	{improbx: choose words with low predictability (with true_pred_prob <= bot x-th percentile)}
#   If no n or x is provided, n defaults to 5, x defaults to 30

#	for all emb_type, use predictions from another emb_type by concat 'emb_type' and 'pred_type':
#	{gpt2-xl-corret: choose words correctly predicted by gpt2}
#	{gpt2-xl-incorret: choose words incorrectly predicted by gpt2}
#	{blenderbot-small-correct: choose words correctly predicted by bbot decoder}
#	{blenderbot-small-incorrect: choose words incorrectly predicted by bbot decoder}
#	{gpt2-pred: choose all words, for words incorrectly predicted by gpt2, use embeddings of the words \
actually predicted by gpt2} (only used for glove embeddings)

# 3. {everything else is purely for the result folder name}

DM := lag2k-25-incorrect
DM := lag10k-25-all
DM := lag2k-25-improb


############## Model Modification ##############
# {best-lag: run encoding using the best lag (lag model with highest correlation)}
# {pc-flip-best-lag: train on comp and test on prod using the best lag model, vice versa}
# {leave empty for regular encoding}
MM := best-lag
MM := pc-flip-best-lag
MM := 

# Choose the command to run: python runs locally, echo is for debugging, sbatch
# is for running on SLURM all lags in parallel.
CMD := echo
CMD := sbatch submit1.sh
CMD := python
# {echo | python | sbatch submit1.sh}

#TODO: move paths to makefile

# plotting modularity
# make separate models with separate electrodes (all at once is possible)
PDIR := $(shell dirname `pwd`)
link-data:
	ln -fs $(PDIR)/247-pickling/results/* data/
	ln -s /projects/HASSON/247/data/podcast-data/*.csv data/
	# ln -fs /scratch/gpfs/${USER}/247-pickling/results/* data/

# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------

# Run the encoding model for the given electrodes in one swoop
# Note that the code will add the subject, embedding type, and PCA details to
# the output folder name
run-encoding:
	mkdir -p logs
	$(CMD) scripts/$(FILE).py \
		--project-id $(PRJCT_ID) \
		--pkl-identifier $(PKL_IDENTIFIER) \
		--datum-emb-fn $(DS) \
		--sid $(SID) \
		--conversation-id $(CONVERSATION_IDX) \
		--electrodes $(E_LIST) \
		--emb-type $(EMB) \
		--context-length $(CNXT_LEN) \
		--align-with $(ALIGN_WITH) \
		--window-size $(WS) \
		--word-value $(WV) \
		--npermutations $(NPERM) \
		--lags $(LAGS) \
		--min-word-freq $(MWF) \
		--fold-num $(FN) \
		--pca-to $(PCA_TO) \
		--layer-idx $(LAYER_IDX) \
		--datum-mod $(DM) \
		--emb-mod $(EM) \
		--model-mod $(MM) \
		$(BC) \
		$(SIG_FN) \
		$(SH) \
		$(PSH) \
		--normalize $(NM)\
		--output-parent-dir $(DT)-$(PRJCT_ID)-$(PKL_IDENTIFIER)-$(SID)-$(EMB)-$(DM)-$(EM) \
		--output-prefix $(USR)-$(WS)ms-$(WV);\


run-encoding-layers:
	mkdir -p logs
	for context in $(CNXT_LEN); do\
		for layer in $(LAYER_IDX); do\
			$(CMD) scripts/$(FILE).py \
				--project-id $(PRJCT_ID) \
				--pkl-identifier $(PKL_IDENTIFIER) \
				--datum-emb-fn $(DS) \
				--sid $(SID) \
				--conversation-id $(CONVERSATION_IDX) \
				--electrodes $(E_LIST) \
				--emb-type $(EMB) \
				--context-length $$context \
				--align-with $(ALIGN_WITH) \
				--window-size $(WS) \
				--word-value $(WV) \
				--npermutations $(NPERM) \
				--lags $(LAGS) \
				--min-word-freq $(MWF) \
				--pca-to $(PCA_TO) \
				--layer-idx $$layer \
				--datum-mod $(DM) \
				--emb-mod $(EM) \
				--model-mod $(MM) \
				$(BC) \
				$(SIG_FN) \
				$(SH) \
				$(PSH) \
				--normalize $(NM)\
				--output-parent-dir $(DT)-$(PRJCT_ID)-$(PKL_IDENTIFIER)-$(SID)-$(EMB)-$(DM)-$(EM)-1024-$$layer \
				--output-prefix $(USR)-$(WS)ms-$(WV);\
		done; \
	done;

# Recommended naming convention for output_folder
#--output-prefix $(USR)-$(WS)ms-$(WV); \

# Run the encoding model for the given electrodes __one at a time__, ideally
# with slurm so it's all parallelized.
run-encoding-slurm:
	mkdir -p logs
	for elec in $(E_LIST); do \
		# for jobid in $(shell seq 1 1); do \
			$(CMD) scripts/$(FILE).py \
				--project-id $(PRJCT_ID) \
				--pkl-identifier $(PKL_IDENTIFIER) \
				--sid $(SID) \
				--electrodes $$elec \
				--conversation-id $(CONVERSATION_IDX) \
				--emb-type $(EMB) \
				--context-length $(CNXT_LEN) \
				--align-with $(ALIGN_WITH) \
				--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
				--window-size $(WS) \
				--word-value $(WV) \
				--npermutations $(NPERM) \
				--lags $(LAGS) \
				--min-word-freq $(MWF) \
				--pca-to $(PCA_TO) \
				$(SH) \
				$(PSH) \
				--normalize $(NM) \
				--output-parent-dir $(PRJCT_ID)-$(PKL_IDENTIFIER)-$(EMB)-pca$(PCA_TO); \
				# --output-prefix ''; \
				# --job-id $(EMB)-$$jobid; \
		# done; \
	done;


# run-sig-encoding-slurm:
# 	mkdir -p logs
# 	for elec in $(E_LIST); do \
# 		# for jobid in $(shell seq 1 1); do \
# 			$(CMD) scripts/$(FILE).py \
# 				--project-id $(PRJCT_ID) \
# 				--pkl-identifier $(PKL_IDENTIFIER) \
# 				--sig-elec-file bobbi.csv \
# 				--emb-type $(EMB) \
# 				--context-length $(CNXT_LEN) \
# 				--align-with $(ALIGN_WITH) \
# 				--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
# 				--window-size $(WS) \
# 				--word-value $(WV) \
# 				--npermutations $(NPERM) \
# 				--lags $(LAGS) \
# 				--min-word-freq $(MWF) \
# 				--pca-to $(PCA_TO) \
# 				$(SH) \
# 				$(PSH) \
# 				--output-parent-dir podcast-gpt2-xl-transcription \
# 				--output-prefix ''; \
# 				# --job-id $(EMB)-$$jobid; \
# 		# done; \
# 	done;


# pca-on-embedding:
# 	python scripts/tfsenc_pca.py \
# 			--sid $(SID) \
# 			--emb-type $(EMB) \
# 			--context-length $(CNXT_LEN) \
# 			--pca-to $(EMB_RED_DIM);


# Run erp for the given electrodes in one swoop

run-erp:
	mkdir -p logs
	$(CMD) scripts/tfserp_main.py \
		--project-id $(PRJCT_ID) \
		--pkl-identifier $(PKL_IDENTIFIER) \
		--datum-emb-fn $(DS) \
		--sid $(SID) \
		--conversation-id $(CONVERSATION_IDX) \
		--electrodes $(E_LIST) \
		--emb-type $(EMB) \
		--context-length $(CNXT_LEN) \
		--align-with $(ALIGN_WITH) \
		--window-size $(WS) \
		--word-value $(WV) \
		--lags $(LAGS) \
		--min-word-freq $(MWF) \
		--layer-idx $(LAYER_IDX) \
		--datum-mod $(DM) \
		--normalize $(NM)\
		$(SIG_FN) \
		--output-parent-dir $(DT)-$(PRJCT_ID)-$(PKL_IDENTIFIER)-$(SID)-erp-$(DM) \
		--output-prefix $(USR)-$(WS)ms-$(WV);\
