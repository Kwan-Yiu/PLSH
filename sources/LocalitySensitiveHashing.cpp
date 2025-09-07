/*
 *   Copyright (c) 2004-2005 Massachusetts Institute of Technology.
 *   All Rights Reserved.
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *   Authors: Alexandr Andoni (andoni@mit.edu), Piotr Indyk (indyk@mit.edu)
 */

/*
  The main functionality of the LSH scheme is in this file (all except
  the hashing of the buckets). This file includes all the functions
  for processing a PRNearNeighborStructT data structure, which is the
  main R-NN data structure based on LSH scheme. The particular
  functions are: initializing a DS, adding new points to the DS, and
  responding to queries on the DS.
 */

#include "headers.h"

void printRNNParameters(FILE *output, RNNParametersT parameters) {
    ASSERT(output != NULL);
    fprintf(output, "R\n");
    fprintf(output, "%0.9lf\n", parameters.parameterR);
    fprintf(output, "Success probability\n");
    fprintf(output, "%0.9lf\n", parameters.successProbability);
    fprintf(output, "Dimension\n");
    fprintf(output, "%d\n", parameters.dimension);
    fprintf(output, "R^2\n");
    fprintf(output, "%0.9lf\n", parameters.parameterR2);
    fprintf(output, "Use <u> functions\n");
    fprintf(output, "%d\n", parameters.useUfunctions);
    fprintf(output, "k\n");
    fprintf(output, "%d\n", parameters.parameterK);
    fprintf(output, "m [# independent tuples of LSH functions]\n");
    fprintf(output, "%d\n", parameters.parameterM);
    fprintf(output, "L\n");
    fprintf(output, "%d\n", parameters.parameterL);
    fprintf(output, "W\n");
    fprintf(output, "%0.9lf\n", parameters.parameterW);
    fprintf(output, "T\n");
    fprintf(output, "%d\n", parameters.parameterT);
    fprintf(output, "typeHT\n");
    fprintf(output, "%d\n", parameters.typeHT);
}

RNNParametersT readRNNParameters(FILE *input) {
    ASSERT(input != NULL);
    RNNParametersT parameters;
    char s[1000];  // TODO: possible buffer overflow

    fscanf(input, "\n");
    fscanf(input, "%[^\n]\n", s);
    FSCANF_REAL(input, &parameters.parameterR);

    fscanf(input, "\n");
    fscanf(input, "%[^\n]\n", s);
    FSCANF_REAL(input, &parameters.successProbability);

    fscanf(input, "\n");
    fscanf(input, "%[^\n]\n", s);
    fscanf(input, "%d", &parameters.dimension);

    fscanf(input, "\n");
    fscanf(input, "%[^\n]\n", s);
    FSCANF_REAL(input, &parameters.parameterR2);

    fscanf(input, "\n");
    fscanf(input, "%[^\n]\n", s);
    fscanf(input, "%d", &parameters.useUfunctions);

    fscanf(input, "\n");
    fscanf(input, "%[^\n]\n", s);
    fscanf(input, "%d", &parameters.parameterK);

    fscanf(input, "\n");
    fscanf(input, "%[^\n]\n", s);
    fscanf(input, "%d", &parameters.parameterM);

    fscanf(input, "\n");
    fscanf(input, "%[^\n]\n", s);
    fscanf(input, "%d", &parameters.parameterL);

    fscanf(input, "\n");
    fscanf(input, "%[^\n]\n", s);
    FSCANF_REAL(input, &parameters.parameterW);

    fscanf(input, "\n");
    fscanf(input, "%[^\n]\n", s);
    fscanf(input, "%d", &parameters.parameterT);

    fscanf(input, "\n");
    fscanf(input, "%[^\n]\n", s);
    fscanf(input, "%d", &parameters.typeHT);

    return parameters;
}

void initHashFunctions(PRNearNeighborStructT nnStruct) {
    ASSERT(nnStruct != NULL);
    LSHFunctionT **lshFunctions;
    FAILIF(NULL == (lshFunctions = (LSHFunctionT **)MALLOC(
                        nnStruct->nHFTuples * sizeof(LSHFunctionT *))));
    for (IntT i = 0; i < nnStruct->nHFTuples; i++) {
        FAILIF(NULL == (lshFunctions[i] = (LSHFunctionT *)MALLOC(
                            nnStruct->hfTuplesLength * sizeof(LSHFunctionT))));
        for (IntT j = 0; j < nnStruct->hfTuplesLength; j++) {
            FAILIF(NULL == (lshFunctions[i][j].a = (RealT *)MALLOC(
                                nnStruct->dimension * sizeof(RealT))));
        }
    }

    for (IntT i = 0; i < nnStruct->nHFTuples; i++) {
        for (IntT j = 0; j < nnStruct->hfTuplesLength; j++) {
            for (IntT d = 0; d < nnStruct->dimension; d++) {
#ifdef USE_L1_DISTANCE
                lshFunctions[i][j].a[d] = genCauchyRandom();
#else
                lshFunctions[i][j].a[d] = genGaussianRandom();
#endif
            }
            // b
            lshFunctions[i][j].b = genUniformRandom(0, nnStruct->parameterW);
        }
    }

    nnStruct->lshFunctions = lshFunctions;
}

PRNearNeighborStructT initializePRNearNeighborFields(
    RNNParametersT algParameters, Int32T nPointsEstimate) {
    PRNearNeighborStructT nnStruct;
    FAILIF(NULL == (nnStruct = (PRNearNeighborStructT)MALLOC(
                        sizeof(RNearNeighborStructT))));
    nnStruct->parameterR = algParameters.parameterR;
    nnStruct->parameterR2 = algParameters.parameterR2;
    nnStruct->useUfunctions = algParameters.useUfunctions;
    nnStruct->parameterK = algParameters.parameterK;
    if (!algParameters.useUfunctions) {
        nnStruct->parameterL = algParameters.parameterL;
        nnStruct->nHFTuples = algParameters.parameterL;
        nnStruct->hfTuplesLength = algParameters.parameterK;
    } else {
        nnStruct->parameterL = algParameters.parameterL;
        nnStruct->nHFTuples = algParameters.parameterM;
        nnStruct->hfTuplesLength = algParameters.parameterK / 2;
    }
    nnStruct->parameterT = algParameters.parameterT;
    nnStruct->dimension = algParameters.dimension;
    nnStruct->parameterW = algParameters.parameterW;

    nnStruct->nPoints = 0;
    nnStruct->pointsArraySize = nPointsEstimate;

    FAILIF(NULL == (nnStruct->points = (PPointT *)MALLOC(
                        nnStruct->pointsArraySize * sizeof(PPointT))));

    initHashFunctions(nnStruct);

    FAILIF(NULL == (nnStruct->pointULSHVectors = (Uns32T **)MALLOC(
                        nnStruct->nHFTuples * sizeof(Uns32T *))));
    for (IntT i = 0; i < nnStruct->nHFTuples; i++) {
        FAILIF(NULL == (nnStruct->pointULSHVectors[i] = (Uns32T *)MALLOC(
                            nnStruct->hfTuplesLength * sizeof(Uns32T))));
    }
    FAILIF(NULL == (nnStruct->precomputedHashesOfULSHs = (Uns32T **)MALLOC(
                        nnStruct->nHFTuples * sizeof(Uns32T *))));
    for (IntT i = 0; i < nnStruct->nHFTuples; i++) {
        FAILIF(NULL ==
               (nnStruct->precomputedHashesOfULSHs[i] = (Uns32T *)MALLOC(
                    N_PRECOMPUTED_HASHES_NEEDED * sizeof(Uns32T))));
    }
    FAILIF(NULL == (nnStruct->reducedPoint =
                        (RealT *)MALLOC(nnStruct->dimension * sizeof(RealT))));
    nnStruct->sizeMarkedPoints = nPointsEstimate;
    FAILIF(NULL == (nnStruct->markedPoints = (BooleanT *)MALLOC(
                        nnStruct->sizeMarkedPoints * sizeof(BooleanT))));
    for (IntT i = 0; i < nnStruct->sizeMarkedPoints; i++) {
        nnStruct->markedPoints[i] = FALSE;
    }
    FAILIF(NULL == (nnStruct->markedPointsIndeces = (Int32T *)MALLOC(
                        nnStruct->sizeMarkedPoints * sizeof(Int32T))));

    nnStruct->reportingResult = TRUE;

    return nnStruct;
}

PRNearNeighborStructT initLSH(RNNParametersT algParameters,
                              Int32T nPointsEstimate) {
    ASSERT(algParameters.typeHT == HT_LINKED_LIST ||
           algParameters.typeHT == HT_STATISTICS);
    PRNearNeighborStructT nnStruct =
        initializePRNearNeighborFields(algParameters, nPointsEstimate);

    FAILIF(NULL == (nnStruct->hashedBuckets = (PUHashStructureT *)MALLOC(
                        nnStruct->parameterL * sizeof(PUHashStructureT))));
    Uns32T *mainHashA = NULL, *controlHash1 = NULL;
    BooleanT uhashesComputedAlready = FALSE;
    for (IntT i = 0; i < nnStruct->parameterL; i++) {
        nnStruct->hashedBuckets[i] = newUHashStructure(
            algParameters.typeHT, nPointsEstimate, nnStruct->parameterK,
            uhashesComputedAlready, mainHashA, controlHash1, NULL);
        uhashesComputedAlready = TRUE;
    }

    return nnStruct;
}

void radix_partition(Int32T *dest, const Int32T *src, Int32T n_items, 
                     Uns32T **precomputedHashes, IntT hash_component_index, IntT n_buckets) {
    Int32T* counts = (Int32T*)CALLOC(n_buckets, sizeof(Int32T));
    for (IntT i = 0; i < n_items; i++) {
        Int32T point_idx = src[i];
        Uns32T bucket_idx = get_precomputed_hash(point_idx, hash_component_index); 
        counts[bucket_idx]++;
    }

    Int32T* offsets = (Int32T*)MALLOC((n_buckets + 1) * sizeof(Int32T));
    offsets[0] = 0;
    for (IntT i = 0; i < n_buckets; i++) {
        offsets[i+1] = offsets[i] + counts[i];
    }

    for (IntT i = 0; i < n_items; i++) {
        Int32T point_idx = src[i];
        Uns32T bucket_idx = get_precomputed_hash(point_idx, hash_component_index);
        dest[offsets[bucket_idx]] = point_idx;
        offsets[bucket_idx]++; 
    }

    FREE(counts);
    FREE(offsets);
}

void preparePointAdding(PRNearNeighborStructT nnStruct, PUHashStructureT uhash,
                        PPointT point);

PRNearNeighborStructT initLSH_WithDataSet(RNNParametersT algParameters,
                                          Int32T nPoints, PPointT *dataSet) {
    ASSERT(algParameters.typeHT == HT_HYBRID_CHAINS);
    ASSERT(dataSet != NULL);
    ASSERT(USE_SAME_UHASH_FUNCTIONS);

    PRNearNeighborStructT nnStruct =
        initializePRNearNeighborFields(algParameters, nPoints);

    nnStruct->nPoints = nPoints;
    for (Int32T i = 0; i < nPoints; i++) {
        nnStruct->points[i] = dataSet[i];
    }

    IntT m = nnStruct->nHFTuples;
    Uns32T **all_u_hashes = (Uns32T**)MALLOC(m * sizeof(Uns32T*));
    for (IntT l = 0; l < m; l++) {
        all_u_hashes[l] = (Uns32T*)MALLOC(nPoints * sizeof(Uns32T));
        for (IntT i = 0; i < nPoints; i++) {
            all_u_hashes[l][i] = compute_u_hash(dataSet[i], nnStruct->lshFunctions[l], nnStruct);
        }
    }

    Int32T* initial_indices = (Int32T*)MALLOC(nPoints * sizeof(Int32T));
    for (Int32T i = 0; i < nPoints; i++) { initial_indices[i] = i; }

    IntT k_half = nnStruct->hfTuplesLength;
    IntT n_buckets_level1 = 1 << k_half; 

    Int32T** level1_partitions = (Int32T**)MALLOC(m * sizeof(Int32T*));
    Int32T** level1_offsets = (Int32T**)MALLOC(m * sizeof(Int32T*));
    for (IntT l = 0; l < m; l++) {
        level1_partitions[l] = (Int32T*)MALLOC(nPoints * sizeof(Int32T));
        level1_offsets[l] = (Int32T*)MALLOC((n_buckets_level1 + 1) * sizeof(Int32T));
        plsh_radix_partition(level1_partitions[l], initial_indices, nPoints, 
                             all_u_hashes, l, n_buckets_level1, level1_offsets[l]);
    }
    FREE(initial_indices);

    nnStruct->hashedBuckets = (PPLSH_HashTableT*)MALLOC(nnStruct->parameterL * sizeof(PPLSH_HashTableT));
    
    IntT firstUComp = 0;
    IntT secondUComp = 1;
    for (IntT i = 0; i < nnStruct->parameterL; i++) {
        nnStruct->hashedBuckets[i] = (PPLSH_HashTableT)MALLOC(sizeof(PLSH_HashTableT));
        PPLSH_HashTableT current_ht = nnStruct->hashedBuckets[i];
        
        current_ht->pointIndices = (Int32T*)MALLOC(nPoints * sizeof(Int32T));
        current_ht->nBuckets = 1 << nnStruct->parameterK; // 2^k
        current_ht->bucketOffsets = (Int32T*)MALLOC((current_ht->nBuckets + 1) * sizeof(Int32T));
        
        Int32T* level1_input = level1_partitions[firstUComp];
        Int32T* level1_offset_info = level1_offsets[firstUComp];
        
        for (IntT bucket_l1 = 0; bucket_l1 < n_buckets_level1; bucket_l1++) {
            Int32T sub_start_pos = level1_offset_info[bucket_l1];
            Int32T sub_num_items = level1_offset_info[bucket_l1+1] - sub_start_pos;

            if (sub_num_items == 0) continue;

            Int32T* src_slice = level1_input + sub_start_pos;
            Int32T* dest_slice = current_ht->pointIndices + sub_start_pos;
            
            // 临时偏移量数组，仅用于本次子分区
            Int32T* sub_offsets_out = (Int32T*)MALLOC((n_buckets_level1 + 1) * sizeof(Int32T));
            
            plsh_radix_partition(dest_slice, src_slice, sub_num_items,
                                 all_u_hashes, secondUComp, n_buckets_level1, sub_offsets_out);

            for(int bucket_l2 = 0; bucket_l2 < n_buckets_level1; bucket_l2++) {
                int final_bucket_idx = (bucket_l1 << k_half) | bucket_l2;
                current_ht->bucketOffsets[final_bucket_idx] = sub_start_pos + sub_offsets_out[bucket_l2];
            }
            FREE(sub_offsets_out);
        }
        current_ht->bucketOffsets[current_ht->nBuckets] = nPoints; 

        secondUComp++;
        if (secondUComp == m) { firstUComp++; secondUComp = firstUComp + 1; }
    }
    
    for (IntT l = 0; l < m; l++) {
        FREE(level1_partitions[l]);
        FREE(level1_offsets[l]);
        FREE(all_u_hashes[l]);
    }
    FREE(level1_partitions);
    FREE(level1_offsets);
    FREE(all_u_hashes);

    return nnStruct;
}

void optimizeLSH(PRNearNeighborStructT nnStruct) {
    ASSERT(nnStruct != NULL);

    PointsListEntryT *auxList = NULL;
    for (IntT i = 0; i < nnStruct->parameterL; i++) {
        optimizeUHashStructure(nnStruct->hashedBuckets[i], auxList);
    }
    FREE(auxList);
}

void freePRNearNeighborStruct(PRNearNeighborStructT nnStruct) {
    if (nnStruct == NULL) {
        return;
    }

    if (nnStruct->points != NULL) {
        free(nnStruct->points);
    }

    if (nnStruct->lshFunctions != NULL) {
        for (IntT i = 0; i < nnStruct->nHFTuples; i++) {
            for (IntT j = 0; j < nnStruct->hfTuplesLength; j++) {
                free(nnStruct->lshFunctions[i][j].a);
            }
            free(nnStruct->lshFunctions[i]);
        }
        free(nnStruct->lshFunctions);
    }

    if (nnStruct->precomputedHashesOfULSHs != NULL) {
        for (IntT i = 0; i < nnStruct->nHFTuples; i++) {
            free(nnStruct->precomputedHashesOfULSHs[i]);
        }
        free(nnStruct->precomputedHashesOfULSHs);
    }

    freeUHashStructure(nnStruct->hashedBuckets[0], TRUE);
    for (IntT i = 1; i < nnStruct->parameterL; i++) {
        freeUHashStructure(nnStruct->hashedBuckets[i], FALSE);
    }
    free(nnStruct->hashedBuckets);

    if (nnStruct->pointULSHVectors != NULL) {
        for (IntT i = 0; i < nnStruct->nHFTuples; i++) {
            free(nnStruct->pointULSHVectors[i]);
        }
        free(nnStruct->pointULSHVectors);
    }

    if (nnStruct->reducedPoint != NULL) {
        free(nnStruct->reducedPoint);
    }

    if (nnStruct->markedPoints != NULL) {
        free(nnStruct->markedPoints);
    }

    if (nnStruct->markedPointsIndeces != NULL) {
        free(nnStruct->markedPointsIndeces);
    }
}

void setResultReporting(PRNearNeighborStructT nnStruct,
                        BooleanT reportingResult) {
    ASSERT(nnStruct != NULL);
    nnStruct->reportingResult = reportingResult;
}

inline void computeULSH(PRNearNeighborStructT nnStruct, IntT gNumber,
                        RealT *point, Uns32T *vectorValue) {
    CR_ASSERT(nnStruct != NULL);
    CR_ASSERT(point != NULL);
    CR_ASSERT(vectorValue != NULL);

    for (IntT i = 0; i < nnStruct->hfTuplesLength; i++) {
        RealT value = 0;
        for (IntT d = 0; d < nnStruct->dimension; d++) {
            value += point[d] * nnStruct->lshFunctions[gNumber][i].a[d];
        }

        vectorValue[i] = (Uns32T)(FLOOR_INT32(
            (value + nnStruct->lshFunctions[gNumber][i].b) /
            nnStruct->parameterW) /* - MIN_INT32T*/);
    }
}

inline void preparePointAdding(PRNearNeighborStructT nnStruct,
                               PUHashStructureT uhash, PPointT point) {
    ASSERT(nnStruct != NULL);
    ASSERT(uhash != NULL);
    ASSERT(point != NULL);

    TIMEV_START(timeComputeULSH);
    for (IntT d = 0; d < nnStruct->dimension; d++) {
        nnStruct->reducedPoint[d] =
            point->coordinates[d] / nnStruct->parameterR;
    }

    for (IntT i = 0; i < nnStruct->nHFTuples; i++) {
        computeULSH(nnStruct, i, nnStruct->reducedPoint,
                    nnStruct->pointULSHVectors[i]);
    }

    if (USE_SAME_UHASH_FUNCTIONS) {
        for (IntT i = 0; i < nnStruct->nHFTuples; i++) {
            precomputeUHFsForULSH(uhash, nnStruct->pointULSHVectors[i],
                                  nnStruct->hfTuplesLength,
                                  nnStruct->precomputedHashesOfULSHs[i]);
        }
    }

    TIMEV_END(timeComputeULSH);
}

void addNewPointToPRNearNeighborStruct(PRNearNeighborStructT nnStruct,
                                       PPointT point) {
    ASSERT(nnStruct != NULL);
    ASSERT(point != NULL);
    ASSERT(nnStruct->reducedPoint != NULL);
    ASSERT(!nnStruct->useUfunctions || nnStruct->pointULSHVectors != NULL);
    ASSERT(nnStruct->hashedBuckets[0]->typeHT == HT_LINKED_LIST ||
           nnStruct->hashedBuckets[0]->typeHT == HT_STATISTICS);

    nnStruct->points[nnStruct->nPoints] = point;
    nnStruct->nPoints++;

    preparePointAdding(nnStruct, nnStruct->hashedBuckets[0], point);

    IntT firstUComp = 0;
    IntT secondUComp = 1;

    TIMEV_START(timeBucketIntoUH);
    for (IntT i = 0; i < nnStruct->parameterL; i++) {
        if (!nnStruct->useUfunctions) {
            addBucketEntry(nnStruct->hashedBuckets[i], 1,
                           nnStruct->precomputedHashesOfULSHs[i], NULL,
                           nnStruct->nPoints - 1);
        } else {
            addBucketEntry(nnStruct->hashedBuckets[i], 2,
                           nnStruct->precomputedHashesOfULSHs[firstUComp],
                           nnStruct->precomputedHashesOfULSHs[secondUComp],
                           nnStruct->nPoints - 1);

            secondUComp++;
            if (secondUComp == nnStruct->nHFTuples) {
                firstUComp++;
                secondUComp = firstUComp + 1;
            }
        }
    }
    TIMEV_END(timeBucketIntoUH);

    if (nnStruct->nPoints > nnStruct->sizeMarkedPoints) {
        nnStruct->sizeMarkedPoints = 2 * nnStruct->nPoints;
        FAILIF(NULL == (nnStruct->markedPoints = (BooleanT *)REALLOC(
                            nnStruct->markedPoints,
                            nnStruct->sizeMarkedPoints * sizeof(BooleanT))));
        for (IntT i = 0; i < nnStruct->sizeMarkedPoints; i++) {
            nnStruct->markedPoints[i] = FALSE;
        }
        FAILIF(NULL == (nnStruct->markedPointsIndeces = (Int32T *)REALLOC(
                            nnStruct->markedPointsIndeces,
                            nnStruct->sizeMarkedPoints * sizeof(Int32T))));
    }
}

inline BooleanT isDistanceSqrLeq(IntT dimension, PPointT p1, PPointT p2,
                                 RealT threshold) {
    RealT result = 0;
    nOfDistComps++;

    TIMEV_START(timeDistanceComputation);
    for (IntT i = 0; i < dimension; i++) {
        RealT temp = p1->coordinates[i] - p2->coordinates[i];
#ifdef USE_L1_DISTANCE
        result += ABS(temp);
#else
        result += SQR(temp);
#endif
        if (result > threshold) {
            return 0;
        }
    }
    TIMEV_END(timeDistanceComputation);

    return 1;
}

Int32T getNearNeighborsFromPRNearNeighborStruct(PRNearNeighborStructT nnStruct,
                                                PPointT query,
                                                PPointT *(&result),
                                                Int32T &resultSize) {
    ASSERT(nnStruct != NULL);
    ASSERT(query != NULL);
    ASSERT(nnStruct->reducedPoint != NULL);
    ASSERT(!nnStruct->useUfunctions || nnStruct->pointULSHVectors != NULL);

    PPointT point = query;

    if (result == NULL) {
        resultSize = RESULT_INIT_SIZE;
        FAILIF(NULL ==
               (result = (PPointT *)MALLOC(resultSize * sizeof(PPointT))));
    }

    preparePointAdding(nnStruct, nnStruct->hashedBuckets[0], point);

    Uns32T precomputedHashesOfULSHs[nnStruct->nHFTuples]
                                   [N_PRECOMPUTED_HASHES_NEEDED];
    for (IntT i = 0; i < nnStruct->nHFTuples; i++) {
        for (IntT j = 0; j < N_PRECOMPUTED_HASHES_NEEDED; j++) {
            precomputedHashesOfULSHs[i][j] =
                nnStruct->precomputedHashesOfULSHs[i][j];
        }
    }
    TIMEV_START(timeTotalBuckets);

    BooleanT oldTimingOn = timingOn;
    if (noExpensiveTiming) {
        timingOn = FALSE;
    }

    IntT firstUComp = 0;
    IntT secondUComp = 1;

    Int32T nNeighbors = 0;     // the number of near neighbors found so far.
    Int32T nMarkedPoints = 0;  // the number of marked points
    for (IntT i = 0; i < nnStruct->parameterL; i++) {
        TIMEV_START(timeGetBucket);
        GeneralizedPGBucket gbucket;
        if (!nnStruct->useUfunctions) {
            gbucket = getGBucket(nnStruct->hashedBuckets[i], 1,
                                 precomputedHashesOfULSHs[i], NULL);
        } else {
            gbucket = getGBucket(nnStruct->hashedBuckets[i], 2,
                                 precomputedHashesOfULSHs[firstUComp],
                                 precomputedHashesOfULSHs[secondUComp]);
            secondUComp++;
            if (secondUComp == nnStruct->nHFTuples) {
                firstUComp++;
                secondUComp = firstUComp + 1;
            }
        }
        TIMEV_END(timeGetBucket);

        PGBucketT bucket;

        TIMEV_START(timeCycleBucket);
        switch (nnStruct->hashedBuckets[i]->typeHT) {
            case HT_LINKED_LIST:
                bucket = gbucket.llGBucket;
                if (bucket != NULL) {
                    PBucketEntryT bucketEntry = &(bucket->firstEntry);
                    while (bucketEntry != NULL) {
                        Int32T candidatePIndex = bucketEntry->pointIndex;
                        PPointT candidatePoint =
                            nnStruct->points[candidatePIndex];
                        if (isDistanceSqrLeq(nnStruct->dimension, point,
                                             candidatePoint,
                                             nnStruct->parameterR2) &&
                            nnStruct->reportingResult) {
                            if (nnStruct->markedPoints[candidatePIndex] ==
                                FALSE) {
                                if (nNeighbors >= resultSize) {
                                    resultSize = 2 * resultSize;
                                    result = (PPointT *)REALLOC(
                                        result, resultSize * sizeof(PPointT));
                                }
                                result[nNeighbors] = candidatePoint;
                                nNeighbors++;
                                nnStruct->markedPointsIndeces[nMarkedPoints] =
                                    candidatePIndex;
                                nnStruct->markedPoints[candidatePIndex] =
                                    TRUE;  
                                nMarkedPoints++;
                            }
                        } else {
                        }
                        bucketEntry = bucketEntry->nextEntry;
                    }
                }
                break;

            case HT_STATISTICS:
                ASSERT(FALSE);  
                break;

            case HT_HYBRID_CHAINS:
                if (gbucket.hybridGBucket != NULL) {
                    PHybridChainEntryT hybridPoint = gbucket.hybridGBucket;
                    Uns32T offset = 0;
                    if (hybridPoint->point.bucketLength == 0) {
                        // there are overflow points in this bucket.
                        offset = 0;
                        for (IntT j = 0; j < N_FIELDS_PER_INDEX_OF_OVERFLOW;
                             j++) {
                            offset += ((Uns32T)((hybridPoint + 1 + j)
                                                    ->point.bucketLength)
                                       << (j * N_BITS_FOR_BUCKET_LENGTH));
                        }
                    }
                    Uns32T index = 0;
                    BooleanT done = FALSE;
                    while (!done) {
                        if (index == MAX_NONOVERFLOW_POINTS_PER_BUCKET) {
                            index = index + offset;
                        }
                        Int32T candidatePIndex =
                            (hybridPoint + index)->point.pointIndex;
                        CR_ASSERT(candidatePIndex >= 0 &&
                                  candidatePIndex < nnStruct->nPoints);
                        done = (hybridPoint + index)->point.isLastPoint == 1
                                   ? TRUE
                                   : FALSE;
                        index++;
                        if (nnStruct->markedPoints[candidatePIndex] == FALSE) {
                            // mark the point first.
                            nnStruct->markedPointsIndeces[nMarkedPoints] =
                                candidatePIndex;
                            nnStruct->markedPoints[candidatePIndex] =
                                TRUE;  
                            nMarkedPoints++;

                            PPointT candidatePoint =
                                nnStruct->points[candidatePIndex];
                            if (isDistanceSqrLeq(nnStruct->dimension, point,
                                                 candidatePoint,
                                                 nnStruct->parameterR2) &&
                                nnStruct->reportingResult) {
                                if (nNeighbors >= resultSize) {
                                    resultSize = 2 * resultSize;
                                    result = (PPointT *)REALLOC(
                                        result, resultSize * sizeof(PPointT));
                                }
                                result[nNeighbors] = candidatePoint;
                                nNeighbors++;
                            }
                        } 
                    }
                }
                break;

            default:
                ASSERT(FALSE);
        }
        TIMEV_END(timeCycleBucket);
    }

    timingOn = oldTimingOn;
    TIMEV_END(timeTotalBuckets);

    // we need to clear the array nnStruct->nearPoints for the next query.
    for (Int32T i = 0; i < nMarkedPoints; i++) {
        ASSERT(nnStruct->markedPoints[nnStruct->markedPointsIndeces[i]] ==
               TRUE);
        nnStruct->markedPoints[nnStruct->markedPointsIndeces[i]] = FALSE;
    }
    DPRINTF("nMarkedPoints: %d\n", nMarkedPoints);

    return nNeighbors;
}
