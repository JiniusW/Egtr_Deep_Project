# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as np

# 최신 NumPy에서 삭제된 옛날 alias들을 복구해서
# bbox 모듈 초기화 시점에 np.float 등이 존재하도록 만든다.
np.float = float
np.int = int
np.bool = bool
np.object = object
np.long = int
try:
    np.bool8 = np.bool_
except AttributeError:
    np.bool8 = bool

DTYPE = np.64float
ctypedef np.64float_t DTYPE_t

def bbox_overlaps(boxes, query_boxes):
    cdef np.ndarray[DTYPE_t, ndim=2] boxes_contig = np.ascontiguousarray(boxes, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] query_contig = np.ascontiguousarray(query_boxes, dtype=DTYPE)

    return bbox_overlaps_c(boxes_contig, query_contig)

cdef np.ndarray[DTYPE_t, ndim=2] bbox_overlaps_c(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bbox_intersections(boxes, query_boxes):
    cdef np.ndarray[DTYPE_t, ndim=2] boxes_contig = np.ascontiguousarray(boxes, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] query_contig = np.ascontiguousarray(query_boxes, dtype=DTYPE)

    return bbox_intersections_c(boxes_contig, query_contig)


cdef np.ndarray[DTYPE_t, ndim=2] bbox_intersections_c(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    For each query box compute the intersection ratio covered by boxes
    ----------
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of intersec between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] intersec = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    intersec[n, k] = iw * ih / box_area
    return intersec