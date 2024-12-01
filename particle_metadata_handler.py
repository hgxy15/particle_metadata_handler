#! /usr/bin/env python
import os
import sys
from cryosparc.tools import CryoSPARC
from cryosparc.dataset import Dataset
import multiprocessing as mp
import optparse
import numpy as np
from typing import List

def rodrigues_vector_to_quaternion(rodrigues_vectors):
    """
    Convert a list of Rodrigues vectors to quaternions.
    input:
        rodrigues_vectors: np.ndarray, shape (N, 3)
    output:
        quaternions: np.ndarray, shape (N, 4)
    """
    theta = np.linalg.norm(rodrigues_vectors, axis=1, keepdims=True)
    theta_half = theta / 2
    small_angle = theta.squeeze() < 1e-8
    
    w = np.cos(theta_half)
    xyz = np.sin(theta_half) / theta * rodrigues_vectors
    
    # Handling small angles where sin(theta/2)/theta approaches 0.5
    # In this case, theta approaches 0, so the Rodrigues vector is a very small rotation
    # division by theta will become a problem, so we will approximate sin(theta/2)/theta by 0.5
    xyz[small_angle,:] = rodrigues_vectors[small_angle,:] * 0.5
    
    quaternions = np.hstack([w, xyz])
    return quaternions

def quaternion_to_rodrigues_vector(quaternions):
    """
    Convert a list of quaternions to Rodrigues vectors.
    input:
        quaternions: np.ndarray, shape (N, 4)
    output:
        rodrigues_vectors: np.ndarray, shape (N, 3)
    """
    # Ensure unit quaternion
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    # check if any of the norms are close to 0, if so, set these to 1,0,0,0
    small_norm = norms.squeeze() < 1e-8
    quaternions = quaternions / norms
    quaternions[small_norm,:] = np.array([1, 0, 0, 0])

    w = quaternions[:, 0]
    xyz = quaternions[:, 1:]

    # Compute theta
    theta = 2 * np.arccos(np.clip(w, -1, 1))  # Clip to handle numerical errors
    sin_half_theta = np.sin(theta / 2)

    # Avoid division by zero for small angles
    small_angle = sin_half_theta < 1e-8
    scale = np.where(small_angle, 2, theta / sin_half_theta)
    
    rodrigues_vectors = xyz * scale[:, np.newaxis]

    return rodrigues_vectors

def perturb_rodrigues_vectors(rodrigues_vectors: np.ndarray,
                            rotation_perturb_range_vec: List[float]) -> np.ndarray:
    """
    Perturb a list of Rodrigues vectors by a given range for the rotation perturbation.
    input:
        rodrigues_vectors: np.ndarray, shape (N, 3)
        rotation_perturb_range_vec: list[float]
    output:
        perturbed_rodrigues_vectors: np.ndarray, shape (N, 3)
    """
    # the norm of the rotation_perturb_range_vec is the amount of rotation in radians.
    perturbation_rotation_amount = np.linalg.norm(rotation_perturb_range_vec)
    # first normalize the rotation_perturb_range_vec
    rotation_perturb_range_vec = np.array(rotation_perturb_range_vec) / perturbation_rotation_amount
    quaternions = rodrigues_vector_to_quaternion(rodrigues_vectors)
    # now generate the perturbation quaternion
    # first select a rotation axis 
    rotation_axis_x = np.random.uniform(-rotation_perturb_range_vec[0], rotation_perturb_range_vec[0], rodrigues_vectors.shape[0])
    rotation_axis_y = np.random.uniform(-rotation_perturb_range_vec[1], rotation_perturb_range_vec[1], rodrigues_vectors.shape[0])
    rotation_axis_z = np.random.uniform(-rotation_perturb_range_vec[2], rotation_perturb_range_vec[2], rodrigues_vectors.shape[0])
    rotation_axis = np.array([rotation_axis_x, rotation_axis_y, rotation_axis_z]).T
    rotation_axis_norm = np.linalg.norm(rotation_axis, axis=1, keepdims=True)
    rotation_axis = rotation_axis / rotation_axis_norm
    
    rotation_amount = np.random.uniform(-perturbation_rotation_amount, perturbation_rotation_amount, rodrigues_vectors.shape[0])
    # now generate the perturbation quaternion
    perturbation_quaternions = rodrigues_vector_to_quaternion(rotation_axis * rotation_amount[:, np.newaxis])
    # now apply the perturbation to the original quaternions
    perturbed_quaternions = quaternion_multiply(perturbation_quaternions, quaternions)
    perturbed_rodrigues_vectors = quaternion_to_rodrigues_vector(perturbed_quaternions)
    return perturbed_rodrigues_vectors

def quaternion_multiply(q1, q2):
    """
    input:
        q1: np.ndarray, shape (N, 4)
        q2: np.ndarray, shape (N, 4)
    output:
        q_product: np.ndarray, shape (N, 4)
    """
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z]).T

def perturb_cryosparc_metadata(input_dataset: Dataset,
                            shift_perturb_range_vec: list[float],
                            rotation_perturb_range_vec: list[float],
                            perturb_location: bool = False,
                            perturb_alignment2D: bool = False,
                            perturb_alignment3D: bool = False) -> Dataset:
    output_dataset = input_dataset.copy()
    if not any([perturb_location, perturb_alignment2D, perturb_alignment3D]):
        raise ValueError("At least one of perturb_location, perturb_alignment2D, or perturb_alignment3D must be True.")
    if perturb_location:
        # check if location is in the metadata
        assert 'location/center_x_frac' in input_dataset.keys(), "location/center_x_frac is not in the input dataset."
        assert 'location/center_y_frac' in input_dataset.keys(), "location/center_y_frac is not in the input dataset."
        assert 'location/micrograph_shape' in input_dataset.keys(), "location/micrograph_shape is not in the input dataset."
        if perturb_alignment2D or perturb_alignment3D:
            raise ValueError("perturb_alignment2D and perturb_alignment3D cannot be True if perturb_location is True.")
        mic_shape_y, mic_shape_x = input_dataset["location/micrograph_shape"].T
        micrograph_psize = input_dataset["blob/psize_A"]
        actual_x_coords = input_dataset["location/center_x_frac"] * mic_shape_x
        actual_y_coords = input_dataset["location/center_y_frac"] * mic_shape_y
        perturbed_x_coords = actual_x_coords + np.random.uniform(-shift_perturb_range_vec[0], shift_perturb_range_vec[0], len(actual_x_coords))
        perturbed_y_coords = actual_y_coords + np.random.uniform(-shift_perturb_range_vec[1], shift_perturb_range_vec[1], len(actual_y_coords))
        output_dataset["location/center_x_frac"] = perturbed_x_coords / mic_shape_x
        output_dataset["location/center_y_frac"] = perturbed_y_coords / mic_shape_y
    elif perturb_alignment2D:
        assert 'alignments2D/pose' in input_dataset.keys(), "alignments2D/pose is not in the input dataset."
        assert 'alignments2D/shift' in input_dataset.keys(), "alignments2D/shift is not in the input dataset."
        if perturb_alignment3D:
            raise ValueError("perturb_alignment2D and perturb_alignment3D cannot be True at the same time.")
        # there is no point in perturbing the in plane rotation alone, as any 2D class average will be equivariant to in plane rotation of particle images.
        # so we will only perturb the shift
        shift_perturbation_array = np.array([np.random.uniform(-shift_perturb_range_vec[0], shift_perturb_range_vec[0], len(input_dataset["alignments2D/shift"][:, 0])), \
                                             np.random.uniform(-shift_perturb_range_vec[1], shift_perturb_range_vec[1], len(input_dataset["alignments2D/shift"][:, 1]))]).T
        perturbed_shift_array = np.array(input_dataset["alignments2D/shift"]) + shift_perturbation_array
        for i in range(perturbed_shift_array.shape[0]):
            output_dataset["alignments2D/shift"][i] = perturbed_shift_array[i, :]
    elif perturb_alignment3D:
        assert 'alignments3D/pose' in input_dataset.keys(), "alignments3D/pose is not in the input dataset."
        assert 'alignments3D/shift' in input_dataset.keys(), "alignments3D/shift is not in the input dataset."
        assert None not in rotation_perturb_range_vec, "rotation_perturb_range_vec cannot contain None when perturb_alignment3D is True."
        shift_perturbation_array = np.array([np.random.uniform(-shift_perturb_range_vec[0], shift_perturb_range_vec[0], len(input_dataset["alignments3D/shift"][:, 0])), \
                                             np.random.uniform(-shift_perturb_range_vec[1], shift_perturb_range_vec[1], len(input_dataset["alignments3D/shift"][:, 1]))]).T
        perturbed_shift_array = np.array(input_dataset["alignments3D/shift"]) + shift_perturbation_array
        original_cryosparc_alignment3D_rotation_array = np.array(input_dataset["alignments3D/pose"])
        # perturb the rotation
        perturbed_cryosparc_alignment3D_rotation_array = perturb_rodrigues_vectors(original_cryosparc_alignment3D_rotation_array, rotation_perturb_range_vec)
        for i in range(perturbed_cryosparc_alignment3D_rotation_array.shape[0]):
            output_dataset["alignments3D/pose"][i] = perturbed_cryosparc_alignment3D_rotation_array[i, :]
            output_dataset["alignments3D/shift"][i] = perturbed_shift_array[i, :]
    else:
        raise ValueError("Invalid perturbation type.")
    
    return output_dataset

def main():
    parser = optparse.OptionParser()
    parser.add_option('-s', '--shift_perturb_range_vec', type='string', help='Shift perturbation range vector')
    parser.add_option('-r', '--rotation_perturb_range_vec', type='string', help='Rotation perturbation range vector')
    parser.add_option('--perturb_location', action='store_true', help='Perturb location')
    parser.add_option('--perturb_alignment2D', action='store_true', help='Perturb alignment2D')
    parser.add_option('--perturb_alignment3D', action='store_true', help='Perturb alignment3D')
    parser.add_option('--project_uid', dest='project_uid', type='str', default=None)
    parser.add_option('--job_uids', dest='job_uids', type='str', default=None)
    # specify base_port
    parser.add_option('--base_port', dest='base_port', type='int', default=39010)
    # email and password.
    parser.add_option('--email', dest='email', type='str', default=None)
    parser.add_option('--password', dest='password', type='str', default=None)
    # env file to load so that we can use the cryosparc python module.
    parser.add_option('--env_file', dest='env_file', type='str', default=None)
    # host and license.
    parser.add_option('--workspace_uid', dest='workspace_uid', type='str', default=None)
    parser.add_option('--host', dest='host', type='str', default=None)
    parser.add_option('--license', dest='license', type='str', default=None)
    (options, args) = parser.parse_args()
    cs = CryoSPARC(host=options.host, \
                    email=options.email, \
                    password=options.password, \
                    base_port=options.base_port)
    project = cs.find_project(options.project_uid)
    job = project.find_job(options.job_uids)
    print("job type: ", job.type)
    sys.stdout.flush()
    if job.type == "select_2D":
        particle_group_name = "particles_selected"
    elif job.type == "particle_sets":
        particle_group_name = "split_0"
    else:
        particle_group_name = "particles"
        print("Warning: infering particle group name from job type: ", job.type)
    particles = job.load_output(particle_group_name)
    print(particles.fields())
    #sys.stdout.flush()
    fields_to_remove = []
    for field in particles.fields():
        if 'filament' in field:
            fields_to_remove.append(field)
        for field in fields_to_remove:
            particles.drop_fields(field)
    input_dataset = particles
    shift_perturb_range_vec = [float(x) for x in options.shift_perturb_range_vec.split(',')]
    rotation_perturb_range_vec = [float(x) for x in options.rotation_perturb_range_vec.split(',')]
    output_dataset = perturb_cryosparc_metadata(input_dataset, shift_perturb_range_vec, rotation_perturb_range_vec, options.perturb_location, options.perturb_alignment2D, options.perturb_alignment3D)
    project.save_external_result(dataset=output_dataset, \
                                    workspace_uid=options.workspace_uid, \
                                    type='particle', \
                                    name='alignment_perturbed_particles', \
                                    title="Alignment_perturbed_particles.")
    
if __name__ == "__main__":
    main()
    
    
