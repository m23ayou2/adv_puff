#include "drone.h"
#include "render.h"

#include <string.h>

#define Env DroneEnv
#define MY_PUT
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    env->num_agents = unpack(kwargs, "num_agents");
    env->max_rings = unpack(kwargs, "max_rings");
    init(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "rings_passed", log->rings_passed);
    assign_to_dict(dict, "ring_collisions", log->ring_collision);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "oob", log->oob);
    assign_to_dict(dict, "timeout", log->timeout);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "n", log->n);
    return 0;
}

static int my_put(Env *env, PyObject *args, PyObject *kwargs) {
    if (kwargs == NULL) {
        return 0;
    }

    PyObject *pred = PyDict_GetItemString(kwargs, "predicted_to_target");
    if (pred != NULL) {
        if (pred == Py_None) {
            memset(env->predicted_to_target, 0, env->num_agents * 3 * sizeof(float));
            return 0;
        }
        if (!PyObject_TypeCheck(pred, &PyArray_Type)) {
            PyErr_SetString(PyExc_TypeError, "predicted_to_target must be a NumPy array");
            return 1;
        }

        PyArrayObject *pred_arr = (PyArrayObject *)pred;
        if (!PyArray_ISCONTIGUOUS(pred_arr)) {
            PyErr_SetString(PyExc_ValueError, "predicted_to_target must be contiguous");
            return 1;
        }
        if (PyArray_TYPE(pred_arr) != NPY_FLOAT32) {
            PyErr_SetString(PyExc_ValueError, "predicted_to_target must be float32");
            return 1;
        }
        if (PyArray_NDIM(pred_arr) != 2 || PyArray_DIM(pred_arr, 1) != 3) {
            PyErr_SetString(PyExc_ValueError, "predicted_to_target must have shape (num_agents, 3)");
            return 1;
        }
        if ((int)PyArray_DIM(pred_arr, 0) != env->num_agents) {
            PyErr_SetString(PyExc_ValueError, "predicted_to_target first dimension must match num_agents");
            return 1;
        }

        memcpy(env->predicted_to_target, PyArray_DATA(pred_arr),
            env->num_agents * 3 * sizeof(float));
    }

    return 0;
}
