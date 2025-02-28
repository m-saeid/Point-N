#!/bin/bash

# Define datasets and (for scanobject) its split modes.
datasets=("modelnet40" "scanobject")
scanobject_modes=("PB_T50_RS")    #("OBJ_BG" "OBJ_ONLY" "PB_T50_RS")

# Define fixed experiment parameters
MODEL="pointgn"
SEED=42
K_VALUES=(100)         # e.g., 90, 100, 110
STAGES=(4)             # e.g., 3, 4, 5
DIMS=(27)              # Both init_dim and stage_dim (e.g., 18, 27, 36)

# Define parameter grids for each function
SIGMAS=(0.35)          # Used for Gaussian-based functions
BLENDS=(0.5)           # Used for functions that use blending
FUSION_METHODS=("variance")  # For adaptive fusion / multihybrid3 (e.g., "variance", "curvature")
EPS_VALUES=(1e-5)      # Epsilon values for numerical stability
CUE_OPTIONS=("curvature")    # For EnrichedGeometricEmbedding (e.g., "curvature", "laplacian")

for dataset in "${datasets[@]}"; do
    if [ "$dataset" == "scanobject" ]; then
        for mode in "${scanobject_modes[@]}"; do
            for DIM in "${DIMS[@]}"; do
                for STAGE in "${STAGES[@]}"; do
                    for K in "${K_VALUES[@]}"; do

                        # EmbeddingHybridAugmented (key: "hybrid")
                        for SIGMA in "${SIGMAS[@]}"; do
                            for BLEND in "${BLENDS[@]}"; do
                                echo "Running: dataset=$dataset mode=$mode, EmbeddingHybridAugmented with sigma=$SIGMA, blend=$BLEND, init_dim=$DIM, stage_dim=$DIM, stages=$STAGE, k=$K"
                                python train_free_main.py --dataset "$dataset" --split "$mode" --model "$MODEL" --embedding_fn "hybrid" \
                                    --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                                    --sigma "$SIGMA" --blend "$BLEND"
                            done
                        done

                        # EmbeddingGaussianLaplacianFusion (key: "gausslap")
                        for SIGMA in "${SIGMAS[@]}"; do
                            for BLEND in "${BLENDS[@]}"; do
                                echo "Running: dataset=$dataset mode=$mode, EmbeddingGaussianLaplacianFusion with sigma=$SIGMA, blend=$BLEND, init_dim=$DIM, stage_dim=$DIM, stages=$STAGE, k=$K"
                                python train_free_main.py --dataset "$dataset" --split "$mode" --model "$MODEL" --embedding_fn "gausslap" \
                                    --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                                    --sigma "$SIGMA" --blend "$BLEND"
                            done
                        done

                        # AdaptiveEmbeddingGPE_1 (key: "adaptive_1")
                        for SIGMA in "${SIGMAS[@]}"; do
                            echo "Running: dataset=$dataset mode=$mode, AdaptiveEmbeddingGPE_1 with sigma=$SIGMA, init_dim=$DIM, stage_dim=$DIM, stages=$STAGE, k=$K"
                            python train_free_main.py --dataset "$dataset" --split "$mode" --model "$MODEL" --embedding_fn "adaptive_1" \
                                --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                                --sigma "$SIGMA"
                        done

                        # AdaptiveEmbeddingGPE_2 (key: "adaptive_2")
                        # The following block is commented out.
                        #: <<'COMMENT'
                        #for SIGMA in "${SIGMAS[@]}"; do
                        #    for FUSION_METHOD in "${FUSION_METHODS[@]}"; do
                        #        for EPS in "${EPS_VALUES[@]}"; do
                        #            echo "Running: dataset=$dataset mode=$mode, AdaptiveEmbeddingGPE_2 with base_sigma=$SIGMA, blend_strategy=$FUSION_METHOD, eps=$EPS, init_dim=$DIM, stage_dim=$DIM, stages=$STAGE, k=$K"
                        #            python train_free_main.py --dataset "$dataset" --split "$mode" --model "$MODEL" --embedding_fn "adaptive_2" \
                        #                --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                        #                --base_sigma "$SIGMA" --blend_strategy "$FUSION_METHOD" --eps "$EPS"
                        #        done
                        #    done
                        #done
                        #COMMENT

                        # EmbeddingMultiHybrid_1 (key: "multihybrid1")
                        for SIGMA in "${SIGMAS[@]}"; do
                            echo "Running: dataset=$dataset mode=$mode, EmbeddingMultiHybrid_1 with sigma=$SIGMA, init_dim=$DIM, stage_dim=$DIM, stages=$STAGE, k=$K"
                            python train_free_main.py --dataset "$dataset" --split "$mode" --model "$MODEL" --embedding_fn "multihybrid1" \
                                --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                                --sigma "$SIGMA"
                        done

                        # EmbeddingMultiHybrid_2 (key: "multihybrid2")
                        for SIGMA in "${SIGMAS[@]}"; do
                            echo "Running: dataset=$dataset mode=$mode, EmbeddingMultiHybrid_2 with sigma=$SIGMA, init_dim=$DIM, stage_dim=$DIM, stages=$STAGE, k=$K"
                            python train_free_main.py --dataset "$dataset" --split "$mode" --model "$MODEL" --embedding_fn "multihybrid2" \
                                --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                                --sigma "$SIGMA"
                        done

                        # EmbeddingMultiHybrid_3 (key: "multihybrid3")
                        # The following block is commented out.
                        #: <<'COMMENT'
                        #for SIGMA in "${SIGMAS[@]}"; do
                        #    for FUSION_METHOD in "${FUSION_METHODS[@]}"; do
                        #        echo "Running: dataset=$dataset mode=$mode, EmbeddingMultiHybrid_3 with sigma=$SIGMA, fusion_method=$FUSION_METHOD, init_dim=$DIM, stage_dim=$DIM, stages=$STAGE, k=$K"
                        #        python train_free_main.py --dataset "$dataset" --split "$mode" --model "$MODEL" --embedding_fn "multihybrid3" \
                        #            --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                        #            --sigma "$SIGMA" --fusion_method "$FUSION_METHOD"
                        #    done
                        #done
                        #COMMENT

                        # EnrichedGeometricEmbedding (key: "geo")
                        for SIGMA in "${SIGMAS[@]}"; do
                            for CUE in "${CUE_OPTIONS[@]}"; do
                                echo "Running: dataset=$dataset mode=$mode, EnrichedGeometricEmbedding with sigma=$SIGMA, cues=$CUE, init_dim=$DIM, stage_dim=$DIM, stages=$STAGE, k=$K"
                                python train_free_main.py --dataset "$dataset" --split "$mode" --model "$MODEL" --embedding_fn "geo" \
                                    --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                                    --sigma "$SIGMA" --cues "$CUE"
                            done
                        done

                    done
                done
            done
        done
    else
        # For modelnet40 (no split mode)
        for DIM in "${DIMS[@]}"; do
            for STAGE in "${STAGES[@]}"; do
                for K in "${K_VALUES[@]}"; do

                    # EmbeddingHybridAugmented (key: "hybrid")
                    for SIGMA in "${SIGMAS[@]}"; do
                        for BLEND in "${BLENDS[@]}"; do
                            echo "Running: dataset=$dataset, EmbeddingHybridAugmented with sigma=$SIGMA, blend=$BLEND, init_dim=$DIM, stage_dim=$DIM, stages=$STAGE, k=$K"
                            python train_free_main.py --dataset "$dataset" --model "$MODEL" --embedding_fn "hybrid" \
                                --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                                --sigma "$SIGMA" --blend "$BLEND"
                        done
                    done

                    # EmbeddingGaussianLaplacianFusion (key: "gausslap")
                    for SIGMA in "${SIGMAS[@]}"; do
                        for BLEND in "${BLENDS[@]}"; do
                            echo "Running: dataset=$dataset, EmbeddingGaussianLaplacianFusion with sigma=$SIGMA, blend=$BLEND, init_dim=$DIM, stage_dim=$DIM, stages=$STAGE, k=$K"
                            python train_free_main.py --dataset "$dataset" --model "$MODEL" --embedding_fn "gausslap" \
                                --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                                --sigma "$SIGMA" --blend "$BLEND"
                        done
                    done

                    # AdaptiveEmbeddingGPE_1 (key: "adaptive_1")
                    for SIGMA in "${SIGMAS[@]}"; do
                        echo "Running: dataset=$dataset, AdaptiveEmbeddingGPE_1 with sigma=$SIGMA, init_dim=$DIM, stage_dim=$DIM, stages=$STAGE, k=$K"
                        python train_free_main.py --dataset "$dataset" --model "$MODEL" --embedding_fn "adaptive_1" \
                            --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                            --sigma "$SIGMA"
                    done

                    # AdaptiveEmbeddingGPE_2 (key: "adaptive_2")
                    # The following block is commented out.
                    #: <<'COMMENT'
                    #for SIGMA in "${SIGMAS[@]}"; do
                    #    for FUSION_METHOD in "${FUSION_METHODS[@]}"; do
                    #        for EPS in "${EPS_VALUES[@]}"; do
                    #            echo "Running: dataset=$dataset, AdaptiveEmbeddingGPE_2 with base_sigma=$SIGMA, blend_strategy=$FUSION_METHOD, eps=$EPS, init_dim=$DIM, stage_dim=$DIM, stages=$STAGE, k=$K"
                    #            python train_free_main.py --dataset "$dataset" --model "$MODEL" --embedding_fn "adaptive_2" \
                    #                --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                    #                --base_sigma "$SIGMA" --blend_strategy "$FUSION_METHOD" --eps "$EPS"
                    #        done
                    #    done
                    #done
                    #COMMENT

                    # EmbeddingMultiHybrid_1 (key: "multihybrid1")
                    for SIGMA in "${SIGMAS[@]}"; do
                        echo "Running: dataset=$dataset, EmbeddingMultiHybrid_1 with sigma=$SIGMA, init_dim=$DIM, stage_dim=$DIM, stages=$STAGE, k=$K"
                        python train_free_main.py --dataset "$dataset" --model "$MODEL" --embedding_fn "multihybrid1" \
                            --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                            --sigma "$SIGMA"
                    done

                    # EmbeddingMultiHybrid_2 (key: "multihybrid2")
                    for SIGMA in "${SIGMAS[@]}"; do
                        echo "Running: dataset=$dataset, EmbeddingMultiHybrid_2 with sigma=$SIGMA, init_dim=$DIM, stage_dim=$DIM, stages=$STAGE, k=$K"
                        python train_free_main.py --dataset "$dataset" --model "$MODEL" --embedding_fn "multihybrid2" \
                            --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                            --sigma "$SIGMA"
                    done

                    # EmbeddingMultiHybrid_3 (key: "multihybrid3")
                    # The following block is commented out.
                    #: <<'COMMENT'
                    #for SIGMA in "${SIGMAS[@]}"; do
                    #    for FUSION_METHOD in "${FUSION_METHODS[@]}"; do
                    #        echo "Running: dataset=$dataset, EmbeddingMultiHybrid_3 with sigma=$SIGMA, fusion_method=$FUSION_METHOD, init_dim=$DIM, stage_dim=$DIM, stages=$STAGE, k=$K"
                    #        python train_free_main.py --dataset "$dataset" --model "$MODEL" --embedding_fn "multihybrid3" \
                    #            --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                    #            --sigma "$SIGMA" --fusion_method "$FUSION_METHOD"
                    #    done
                    #done
                    #COMMENT

                    # EnrichedGeometricEmbedding (key: "geo")
                    for SIGMA in "${SIGMAS[@]}"; do
                        for CUE in "${CUE_OPTIONS[@]}"; do
                            echo "Running: dataset=$dataset, EnrichedGeometricEmbedding with sigma=$SIGMA, cues=$CUE, init_dim=$DIM, stage_dim=$DIM, stages=$STAGE, k=$K"
                            python train_free_main.py --dataset "$dataset" --model "$MODEL" --embedding_fn "geo" \
                                --seed "$SEED" --k "$K" --init_dim "$DIM" --stage_dim "$DIM" --stages "$STAGE" \
                                --sigma "$SIGMA" --cues "$CUE"
                        done
                    done

                done
            done
        done
    fi
done
