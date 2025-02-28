#!/bin/bash
# Define lists of hyperparameters.
embedding_functions=("sine" "cosine" "tanh" "poly" "lap" "recip" "sinc" "gpe")
# Best performing values from CSV: dims=64, stages=5, k=90.
dims=(18 27 36) # 45)               # Both init_dim and stage_dim are set to 64.
stages=(3 4 5)              # Number of stages.
ks=(90 100 110)  #(80 90 100 110 120 130)                 # Neighborhood size.
# Sigma values are only used by EmbeddingGPE and EmbeddingLaplacian.
sigmas=(0.3 0.35 0.4)  # (0.25 0.3 0.35 0.4 0.45)

# Define datasets and, for scanobject, its split modes.
datasets=("modelnet40" "scanobject")
scanobject_modes=("OBJ_BG" "OBJ_ONLY" "PB_T50_RS")

for dataset in "${datasets[@]}"; do
    if [ "$dataset" == "scanobject" ]; then
        for mode in "${scanobject_modes[@]}"; do
            for embedding in "${embedding_functions[@]}"; do
                for dim in "${dims[@]}"; do
                    for stage in "${stages[@]}"; do
                        for k in "${ks[@]}"; do
                            if [[ "$embedding" == "gpe" || "$embedding" == "lap" ]]; then
                                for sigma in "${sigmas[@]}"; do
                                    echo "Running: dataset=$dataset mode=$mode embedding_fn=$embedding init_dim=$dim stage_dim=$dim stages=$stage k=$k sigma=$sigma"
                                    python train_free_main.py --dataset "$dataset" --split "$mode" --embedding_fn "$embedding" --init-dim "$dim" --stage-dim "$dim" --stages "$stage" --k "$k" --sigma "$sigma"
                                done
                            else
                                echo "Running: dataset=$dataset mode=$mode embedding_fn=$embedding init_dim=$dim stage_dim=$dim stages=$stage k=$k"
                                python train_free_main.py --dataset "$dataset" --split "$mode" --embedding_fn "$embedding" --init-dim "$dim" --stage-dim "$dim" --stages "$stage" --k "$k"
                            fi
                        done
                    done
                done
            done
        done
    else
        for embedding in "${embedding_functions[@]}"; do
            for dim in "${dims[@]}"; do
                for stage in "${stages[@]}"; do
                    for k in "${ks[@]}"; do
                        if [[ "$embedding" == "gpe" || "$embedding" == "lap" ]]; then
                            for sigma in "${sigmas[@]}"; do
                                echo "Running: dataset=$dataset embedding_fn=$embedding init_dim=$dim stage_dim=$dim stages=$stage k=$k sigma=$sigma"
                                python train_free_main.py --dataset "$dataset" --embedding_fn "$embedding" --init-dim "$dim" --stage-dim "$dim" --stages "$stage" --k "$k" --sigma "$sigma"
                            done
                        else
                            echo "Running: dataset=$dataset embedding_fn=$embedding init_dim=$dim stage_dim=$dim stages=$stage k=$k"
                            python train_free_main.py --dataset "$dataset" --embedding_fn "$embedding" --init-dim "$dim" --stage-dim "$dim" --stages "$stage" --k "$k"
                        fi
                    done
                done
            done
        done
    fi
done


#########################################################
#########################################################
#########################################################
#########################################################


#!/bin/bash

# Define fixed experiment parameters
#DATASET="modelnet40"    # Choose between "modelnet40" and "scanobject"
#MODEL="pointgn"
#SEED=42
#K_VALUES=(90 100 110)  # Different k values to test
#STAGES=(3 4 5)         # Number of stages
#DIMS=(18 27 36)        # Both init_dim and stage_dim are set to the same values

# Define parameter grids for each function
#SIGMAS=(0.3 0.35 0.4)             # Used for Gaussian-based functions
#BLENDS=(0.4 0.5 0.6)              # Used for functions that use blending
#FUSION_METHODS=("variance" "curvature")  # Used in fusion-related functions
#EPS_VALUES=(1e-5 1e-6)            # Epsilon values for numerical stability
#CUE_OPTIONS=("curvature" "laplacian")  # For EnrichedGeometricEmbedding

# Run experiments for different embedding functions
#!/bin/bash

# Define fixed experiment parameters
DATASET="modelnet40"    # Choose between "modelnet40" and "scanobject"
MODEL="pointgn"
SEED=42
K_VALUES=(100)            # Different k values to test (e.g., 90 100 110)
STAGES=(4)                # Number of stages (e.g., 3 4 5)
DIMS=(27)                 # Both init_dim and stage_dim are set to the same values (e.g., 18 27 36)

# Define parameter grids for each function
SIGMAS=(0.35)             # Used for Gaussian-based functions
BLENDS=(0.5)              # Used for functions that use blending
FUSION_METHODS=("variance")  # Used in adaptive blending and for multihybrid3 (e.g., "variance" "curvature")
EPS_VALUES=(1e-5)         # Epsilon values for numerical stability
CUE_OPTIONS=("curvature") # For EnrichedGeometricEmbedding (e.g., "curvature" "laplacian")

for DIM in "${DIMS[@]}"; do
    for STAGE in "${STAGES[@]}"; do
        for K in "${K_VALUES[@]}"; do


            # EmbeddingHybridAugmented (key: "hybrid")
            for SIGMA in "${SIGMAS[@]}"; do
                for BLEND in "${BLENDS[@]}"; do
                    echo "Running: EmbeddingHybridAugmented with sigma=$SIGMA, blend=$BLEND"
                    python train_free_main.py --dataset "$DATASET" --model "$MODEL" --embedding_fn "hybrid" \
                        --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                        --sigma "$SIGMA" --blend "$BLEND"
                done
            done

            # EmbeddingGaussianLaplacianFusion (key: "gausslap")
            for SIGMA in "${SIGMAS[@]}"; do
                for BLEND in "${BLENDS[@]}"; do
                    echo "Running: EmbeddingGaussianLaplacianFusion with sigma=$SIGMA, blend=$BLEND"
                    python train_free_main.py --dataset "$DATASET" --model "$MODEL" --embedding_fn "gausslap" \
                        --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                        --sigma "$SIGMA" --blend "$BLEND"
                done
            done

            # AdaptiveEmbeddingGPE_1 (key: "adaptive_1")
            for SIGMA in "${SIGMAS[@]}"; do
                echo "Running: AdaptiveEmbeddingGPE_1 with sigma=$SIGMA"
                python train_free_main.py --dataset "$DATASET" --model "$MODEL" --embedding_fn "adaptive_1" \
                    --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                    --sigma "$SIGMA"
            done

            ### AdaptiveEmbeddingGPE_2 (key: "adaptive_2")
            ## for SIGMA in "${SIGMAS[@]}"; do
            ##    for FUSION_METHOD in "${FUSION_METHODS[@]}"; do
            ##        for EPS in "${EPS_VALUES[@]}"; do
            ##            echo "Running: AdaptiveEmbeddingGPE_2 with base_sigma=$SIGMA, blend_strategy=$FUSION_METHOD, eps=$EPS"
            ##            python train_free_main.py --dataset "$DATASET" --model "$MODEL" --embedding_fn "adaptive_2" \
            ##                --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
            ##                --base_sigma "$SIGMA" --blend_strategy "$FUSION_METHOD" --eps "$EPS"
            ##        done
            ##    done
            ##done

            # EmbeddingMultiHybrid_1 (key: "multihybrid1")
            for SIGMA in "${SIGMAS[@]}"; do
                echo "Running: EmbeddingMultiHybrid_1 with sigma=$SIGMA"
                python train_free_main.py --dataset "$DATASET" --model "$MODEL" --embedding_fn "multihybrid1" \
                    --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                    --sigma "$SIGMA"
            done

            # EmbeddingMultiHybrid_2 (key: "multihybrid2")
            for SIGMA in "${SIGMAS[@]}"; do
                echo "Running: EmbeddingMultiHybrid_2 with sigma=$SIGMA"
                python train_free_main.py --dataset "$DATASET" --model "$MODEL" --embedding_fn "multihybrid2" \
                    --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                    --sigma "$SIGMA"
            done

            ## EmbeddingMultiHybrid_3 (key: "multihybrid3")
            ##for SIGMA in "${SIGMAS[@]}"; do
            ##    for FUSION_METHOD in "${FUSION_METHODS[@]}"; do
            ##        echo "Running: EmbeddingMultiHybrid_3 with sigma=$SIGMA, fusion_method=$FUSION_METHOD"
            ##        python train_free_main.py --dataset "$DATASET" --model "$MODEL" --embedding_fn "multihybrid3" \
            ##            --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
            ##            --sigma "$SIGMA" --fusion_method "$FUSION_METHOD"
            ##    done
            ##done

            # EnrichedGeometricEmbedding (key: "geo")
            for SIGMA in "${SIGMAS[@]}"; do
                for CUE in "${CUE_OPTIONS[@]}"; do
                    echo "Running: EnrichedGeometricEmbedding with sigma=$SIGMA, cues=$CUE"
                    python train_free_main.py --dataset "$DATASET" --model "$MODEL" --embedding_fn "geo" \
                        --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                        --sigma "$SIGMA" --cues "$CUE"
                done
            done

        done
    done
done
