#define checkCudaError( cudaError ) __checkCudaError( cudaError, __FILE__, __LINE__ )

#include <Eigen/Eigen>
#include <cuda_runtime.h>
#include <stdio.h>
#include <ctime>
#include <unistd.h>
#include <opencv2/photo/photo.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#define MAX_BLOCK 400
#define THREAD_NUM 256
#define CLOUD_NUM 65536
#define GRID_NUM 100000
#define SO2_NUM 36

using namespace Eigen;
using namespace std;

// Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> esolver(ATA);
// Eigen::Matrix<float, 1, 6> matE;
// Eigen::Matrix<float, 6, 6> matV;
// matE = esolver.eigenvalues().real();
// matV = esolver.eigenvectors().real();

void __checkCudaError ( cudaError_t result_t, const char * file, const int line )
{
    std::string error_string;

    if ( cudaSuccess != result_t && cudaErrorCudartUnloading != result_t )
    {
        fprintf ( stderr, "\x1B[31m CUDA error encountered in file '%s', line %d\n Error %d: %s\n Terminating FIRE!\n \x1B[0m", file, line, result_t,
               cudaGetErrorString ( result_t ) );
        printf("CUDA error encountered: %s", cudaGetErrorString ( result_t ) );
        printf(". Terminating application.\n");
        throw std::runtime_error ( "checkCUDAError : ERROR: CUDA Error" );
    }
}

// map param
__constant__ int   size_pos[2];
__constant__ int   size_yaw;
__constant__ int   xy_num;
__constant__ int   xyyaw_num;
__constant__ int   inpaint_method;
__constant__ float length_pos[2];
__constant__ float length_yaw;
__constant__ float resolution_pos;
__constant__ float resolution_pos_inv;
__constant__ float resolution_yaw;
__constant__ float resolution_yaw_inv;
__constant__ float origin[2];
__constant__ float min_var;
__constant__ float max_var;
__constant__ float hori_var;
__constant__ float mahalanobis_threshold;
__constant__ float normal_radius;
__constant__ float min_cosxi;
__constant__ float max_curvature;
__constant__ float weight_xi;
__constant__ float weight_curvature;
__constant__ float ignore_z_min;
__constant__ float ignore_z_max;
__constant__ float beam_sigma2;
__constant__ float body2sensor_T[3];

// map data
__device__ bool map_nan[GRID_NUM];
__device__ float elevation[GRID_NUM];
__device__ float var[GRID_NUM];
__device__ float inpainted[GRID_NUM];
__device__ float smooth[GRID_NUM];
__device__ float normal_x[GRID_NUM];
__device__ float normal_y[GRID_NUM];
__device__ float risk[GRID_NUM];
__device__ float sdf[GRID_NUM];

// cloud temp
__device__ int temp_idx[CLOUD_NUM];
__device__ float temp_var[CLOUD_NUM];
__device__ float temp_height[CLOUD_NUM];

__host__ __device__ void wrapIndexOne(int& idx, int buffer_size)
{
    if (idx < buffer_size)
    {
        if(idx >= 0)
            return;
        else if(idx >= -buffer_size)
        {
            idx += buffer_size;
            return;
        }
        else
        {
            idx = idx % buffer_size;
            idx += buffer_size;
        }
    }
    else if(idx < buffer_size*2)
    {
        idx -= buffer_size;
        return;
    }
    else
        idx = idx % buffer_size;
}

__device__ int signFunc(const int val)
{
    return static_cast<int>(0 < val) - static_cast<int>(val < 0);
} 

__device__ void boundPos(Vector2f& pos, const Vector2f& position)
{
    Vector2f positionShifted = pos - position;
    positionShifted[0] += origin[0];
    positionShifted[1] += origin[1];

    for (int i = 0; i < positionShifted.size(); i++)
    {
        float epsilon = 10.0 * 1e-10;
        if (fabs(pos(i)) > 1.0)
            epsilon *= fabs(pos(i));

        if (positionShifted(i) <= 0)
        {
            positionShifted(i) = epsilon;
            continue;
        }
        if (positionShifted(i) >= length_pos[i])
        {
            positionShifted(i) = length_pos[i] - epsilon;
            continue;
        }
    }

    pos = positionShifted + position;
    pos[0] -= origin[0];
    pos[1] -= origin[1];
}


__device__ void computerEigenvalue(float *pMatrix, int nDim, float *maxvector, float *curvature, float dbEps, int nJt)
{
    float pdblVects[9];
    float pdbEigenValues[3];
    
	for(int i = 0; i < nDim; i ++) 
	{   
		pdblVects[i*nDim+i] = 1.0f; 
		for(int j = 0; j < nDim; j ++) 
		{ 
			if(i != j)   
				pdblVects[i*nDim+j]=0.0f; 
		} 
	} 
 
	int nCount = 0;
	while(1)
	{
		float dbMax = pMatrix[1];
		int nRow = 0;
		int nCol = 1;
		for (int i = 0; i < nDim; i ++)
		{
			for (int j = 0; j < nDim; j ++)
			{
				float d = fabs(pMatrix[i*nDim+j]); 
 
				if((i!=j) && (d> dbMax)) 
				{ 
					dbMax = d;   
					nRow = i;   
					nCol = j; 
				} 
			}
		}
 
		if(dbMax < dbEps) 
			break;  
 
		if(nCount > nJt)
			break;
 
		nCount++;
 
		float dbApp = pMatrix[nRow*nDim+nRow];
		float dbApq = pMatrix[nRow*nDim+nCol];
		float dbAqq = pMatrix[nCol*nDim+nCol];
 
		float dbAngle = 0.5*atan2(-2*dbApq,dbAqq-dbApp);
		float dbSinTheta = sin(dbAngle);
		float dbCosTheta = cos(dbAngle);
		float dbSin2Theta = sin(2*dbAngle);
		float dbCos2Theta = cos(2*dbAngle);
 
		pMatrix[nRow*nDim+nRow] = dbApp*dbCosTheta*dbCosTheta + 
			dbAqq*dbSinTheta*dbSinTheta + 2*dbApq*dbCosTheta*dbSinTheta;
		pMatrix[nCol*nDim+nCol] = dbApp*dbSinTheta*dbSinTheta + 
			dbAqq*dbCosTheta*dbCosTheta - 2*dbApq*dbCosTheta*dbSinTheta;
		pMatrix[nRow*nDim+nCol] = 0.5*(dbAqq-dbApp)*dbSin2Theta + dbApq*dbCos2Theta;
		pMatrix[nCol*nDim+nRow] = pMatrix[nRow*nDim+nCol];
 
		for(int i = 0; i < nDim; i ++) 
		{ 
			if((i!=nCol) && (i!=nRow)) 
			{ 
				int u = i*nDim + nRow;	//p  
				int w = i*nDim + nCol;	//q
				dbMax = pMatrix[u]; 
				pMatrix[u]= pMatrix[w]*dbSinTheta + dbMax*dbCosTheta; 
				pMatrix[w]= pMatrix[w]*dbCosTheta - dbMax*dbSinTheta; 
			} 
		} 
 
		for (int j = 0; j < nDim; j ++)
		{
			if((j!=nCol) && (j!=nRow)) 
			{ 
				int u = nRow*nDim + j;	//p
				int w = nCol*nDim + j;	//q
				dbMax = pMatrix[u]; 
				pMatrix[u]= pMatrix[w]*dbSinTheta + dbMax*dbCosTheta; 
				pMatrix[w]= pMatrix[w]*dbCosTheta - dbMax*dbSinTheta; 
			} 
		}
 
		for(int i = 0; i < nDim; i ++) 
		{ 
			int u = i*nDim + nRow;		//p   
			int w = i*nDim + nCol;		//q
			dbMax = pdblVects[u]; 
			pdblVects[u] = pdblVects[w]*dbSinTheta + dbMax*dbCosTheta; 
			pdblVects[w] = pdblVects[w]*dbCosTheta - dbMax*dbSinTheta; 
		} 
 
	}
    
    int min_id = 0;
	float minEigenvalue;
    float sumEigenvalue = 0.0;

	for(int i = 0; i < nDim; i ++) 
	{   
		pdbEigenValues[i] = pMatrix[i*nDim+i];
        sumEigenvalue += pdbEigenValues[i];
        if(i == 0)
            minEigenvalue = pdbEigenValues[i];
        else
        {
            if(minEigenvalue > pdbEigenValues[i])
            {
                minEigenvalue = pdbEigenValues[i];
                min_id = i;	
            }
        }
    } 

    for(int i = 0; i < nDim; i ++) 
    {  
        maxvector[i] = pdblVects[min_id + nDim * i];
    }

    *curvature = 3.0 * minEigenvalue / sumEigenvalue;
}

__global__ void init_map()
{
    int map_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (map_idx < xy_num)
    {
        map_nan[map_idx] = true;
        sdf[map_idx] = 0.0;
    }

    //! xulong
    // if (map_idx == 0)
    // {
    //     printf("device params:\n");
    //     printf("size_pos[0] = %d\n", size_pos[0]);
    //     printf("size_pos[1] = %d\n", size_pos[1]);
    //     printf("size_yaw = %d\n", size_yaw);
    //     printf("xy_num = %d\n", xy_num);
    //     printf("xyyaw_num = %d\n", xyyaw_num);
    //     printf("resolution_pos_inv = %f\n", resolution_pos_inv);
    //     printf("resolution_yaw_inv = %f\n", resolution_yaw_inv);
    //     printf("resolution_pos = %f\n", resolution_pos);
    //     printf("resolution_yaw = %f\n", resolution_yaw);
    //     printf("length_pos[0] = %f\n", length_pos[0]);
    //     printf("length_pos[1] = %f\n", length_pos[1]);
    //     printf("length_yaw =6.284\n");
    //     printf("origin[0] = %f\n", origin[0]);
    //     printf("origin[1] = %f\n", origin[1]);
    //     printf("min_var = %f\n", min_var);
    //     printf("max_var = %f\n", max_var);
    //     printf("hori_var = %f\n", hori_var);
    //     printf("mahalanobis_threshold, %f\n", mahalanobis_threshold);
    //     printf("normal_radius = %f\n", normal_radius);
    //     printf("min_cosxi = %f\n", min_cosxi);
    //     printf("max_curvature = %f\n", max_curvature);
    //     printf("weight_xi = %f\n", weight_xi);
    //     printf("weight_curvature = %f\n", weight_curvature);
    //     printf("ignore_z_min = %f\n", ignore_z_min);
    //     printf("ignore_z_max = %f\n", ignore_z_max);
    //     printf("body2sensor_T[0] = %f\n", body2sensor_T[0]);
    //     printf("body2sensor_T[1] = %f\n", body2sensor_T[1]);
    //     printf("body2sensor_T[2] = %f\n", body2sensor_T[2]);
    //     printf("beam_sigma2 = %f\n", beam_sigma2);
    // }
}

__global__ void clear_map(Vector2i clear_idx, 
                        Vector2i clear_nums, 
                        Vector2i clear_idx2,
                        Vector2i clear_nums2,
                        bool clear_all,
                        bool clear_two)
{
    int map_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int size_row = size_pos[0];
    int size_col = size_pos[1];

    if (clear_all)
    {
        if (map_idx < xy_num)
            map_nan[map_idx] = true;
    }

    else if (map_idx < xy_num)
    {
        int clear_rows = clear_nums[0];
        int clear_cols = clear_nums[1];
        int cell_num_row = size_col * clear_rows;
        int cell_num_col = size_row * clear_cols;
        if (map_idx < cell_num_row)
            map_nan[map_idx / clear_rows * size_row + map_idx % clear_rows + clear_idx[0]] = true;
        if (map_idx < cell_num_col)
            map_nan[clear_idx[1] * size_row + map_idx] = true;
        if (clear_two)
        {
            int clear_rows2 = clear_nums2[0];
            int clear_cols2 = clear_nums2[1];
            int cell_num_row2 = size_col * clear_rows2;
            int cell_num_col2 = size_row * clear_cols2;
            if (map_idx < cell_num_row2)
                map_nan[map_idx / clear_rows2 * size_row + map_idx % clear_rows2 + clear_idx2[0]] = true;
            if (map_idx < cell_num_col2)
                map_nan[clear_idx2[1] * size_row + map_idx] = true;
        }
    }
}

__global__ void compute_var(int point_num,
                            float* dev_points, 
                            Vector3f world_T,
                            Matrix3f body_R,
                            Matrix3f body_Cov_R,
                            float body_Cov_Z,
                            Array2i start_index,
                            Vector2f position)
{
    // deal with cloud var
    const int pt_num = point_num;
    const float ignore_z_min_ = ignore_z_min;
    const float ignore_z_max_ = ignore_z_max;
    const float beam_sigma2_ = beam_sigma2;
    const float resolution_pos_inv_ = resolution_pos_inv;

    // check if in num range
    int pt_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (pt_idx < pt_num)
    {
        Vector3f point_world(dev_points[pt_idx*3], dev_points[pt_idx*3+1], dev_points[pt_idx*3+2]);
        Map<Vector2f> origin_(origin);

        // check if in map range
        Vector2f pos_map = origin_ + position - point_world.head(2);
        if (pos_map.x() >= 0.0 && pos_map.y() >= 0.0 && \
            pos_map.x() < length_pos[0] && pos_map.y() < length_pos[1])
        {
            // check if in height range
            float height_body = body_R.col(2).dot(point_world) + world_T[2];
            if (height_body > ignore_z_min_ && height_body < ignore_z_max_)
            {
                // compute var
                Vector3f point_sensor;
                point_sensor[0] = body_R.col(0).dot(point_world) + world_T[0] + body2sensor_T[0];
                point_sensor[1] = body_R.col(1).dot(point_world) + world_T[1] + body2sensor_T[1];
                point_sensor[2] = height_body + body2sensor_T[2];

                Vector3f Js = body_R.row(2);
                Vector3f Jr = Vector3f::Zero();
                Vector3f SigmaS = Vector3f::Ones();

                Vector3f point_sensor2 = point_sensor.cwiseAbs2();
                SigmaS = point_sensor2 / point_sensor2.sum() * beam_sigma2_;

                Vector3f dVec = point_sensor;
                dVec[0] -= body2sensor_T[0];
                dVec[1] -= body2sensor_T[1];
                dVec[2] -= body2sensor_T[2];
                Jr[0] = -dVec[2] * Js[1] + dVec[1] * Js[2];
                Jr[1] = dVec[2] * Js[0] - dVec[0] * Js[2];
                Jr[2] = -dVec[1] * Js[0] + dVec[0] * Js[1];
                Vector3f sJr;
                sJr[0] = body_Cov_R.row(0).dot(Jr);
                sJr[1] = body_Cov_R.row(1).dot(Jr);
                sJr[2] = body_Cov_R.row(2).dot(Jr);
                float variance = body_Cov_Z
                                + (SigmaS.cwiseProduct(Js)).dot(Js)
                                + sJr.dot(Jr);

                // add to temp
                Array2i index;
                Array2f indexf = pos_map.array() * resolution_pos_inv_;
                index(0) = (int)indexf(0);
                index(1) = (int)indexf(1);
                if ( start_index[0] != 0 || start_index[1] != 0)
                {
                    index[0] += start_index[0];
                    index[1] += start_index[1];
                    for (int i=0; i<2; i++)
                        wrapIndexOne(index[i], size_pos[i]);
                }
                int true_idx = index[1] * size_pos[0] + index[0];
                temp_height[pt_idx] = point_world.z();
                temp_idx[pt_idx] = true_idx;
                temp_var[pt_idx] = variance;
            }
            else
                temp_idx[pt_idx] = -1;
        }
        else
            temp_idx[pt_idx] = -1;
    }
}

__global__ void compute_elevation(int point_num)
{
    // add to map ( var update using KF )
    int map_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (map_idx < xy_num)
    {
        sdf[map_idx] = 0.0;
        risk[map_idx] = 0.0;
        for (int i=0; i<point_num; i++)
        {
            int true_idx = temp_idx[i];
            float height = temp_height[i];
            float tvar = temp_var[i];
            if (true_idx == -1 || true_idx != map_idx)
                continue;

            if (map_nan[map_idx])
            {
                elevation[map_idx] = height;
                var[map_idx] = tvar;
                map_nan[map_idx] = false;
            }else
            {
                float mahalanobisDistance = fabs(height - elevation[map_idx]) / sqrt(tvar);
                if (mahalanobisDistance > mahalanobis_threshold)
                {
                    elevation[map_idx] = height;
                    var[map_idx] = tvar;
                }
                else
                {
                    float var_old = var[map_idx];
                    float ele_old = elevation[map_idx];
                    elevation[map_idx] = (var_old * height + tvar * ele_old)
                                                        / (var_old + tvar);
                    var[map_idx] = (var_old * tvar) / (var_old + tvar);
                }
            }
            var[map_idx] = max(min_var, var[map_idx]);
        }
        if (map_nan[map_idx])
        {
            elevation[map_idx] = NAN;
            var[map_idx] = NAN;
        }
    }
}

__global__ void compute_inpaint_spiral(Array2i start_index,
                                        Vector2f position)
{
    int map_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (map_idx < xy_num)
    {
        int map_idx_true = map_idx / size_pos[1] + (map_idx % size_pos[1]) * size_pos[0];
        if (inpaint_method != 1)
        {
            inpainted[map_idx_true] = elevation[map_idx_true];
            return;
        }
        if (map_nan[map_idx_true])
        {
            inpainted[map_idx_true] = NAN;
            Array2i idx(map_idx/size_pos[1], map_idx%size_pos[1]);
            if ( start_index[0] != 0 || start_index[1] != 0)
            {
                idx[0] -= start_index[0];
                idx[1] -= start_index[1];
                for (int i=0; i<2; i++)
                    wrapIndexOne(idx[i], size_pos[i]);
            }

            float radius2 = length_pos[0] * length_pos[0] + length_pos[1] * length_pos[1];
            int n_rings = (int) (sqrt(radius2) / resolution_pos) + 1;
            int distance_ = 0;
            do
            {
                bool get_nearest = false;
                distance_++;
                Array2i point(distance_, 0);
                Array2i pointInMap;
                Array2i normal;
                do
                {
                    pointInMap = point + idx;
                    if (pointInMap[0] >= 0 && pointInMap[1] >= 0 &&
                        pointInMap[0] < size_pos[0] && pointInMap[1] < size_pos[1])
                    {
                         pointInMap += start_index;
                        for (int i=0; i<2; i++)
                            wrapIndexOne(pointInMap[i], size_pos[i]);
                        // col-major
                        int true_idx = pointInMap[1] * size_pos[0] + pointInMap[0];
                        if (!map_nan[true_idx])
                        {
                            get_nearest = true;
                            inpainted[map_idx_true] = elevation[true_idx];
                            break;
                        }
                    }

                    normal[0] = -signFunc(point[1]);
                    normal[1] = signFunc(point[0]);
                    if (normal[0] !=0 && (int)(Vector2f(point[0] + normal[0], point[1]).norm()) == distance_)
                        point[0] += normal[0];
                    if (normal[0] !=0 && (int)(Vector2f(point[0], point[1] + normal[1]).norm()) == distance_)
                        point[1] += normal[1];
                    else
                    {
                        point += normal;
                    }
                } while (point[0]!=distance_ || point[1]!=0);
                if (get_nearest)
                    break;
            }while(distance_<n_rings);
        }
        else
        {
            inpainted[map_idx_true] = elevation[map_idx_true];
        }
    }
}

__global__ void compute_map_r2(Array2i start_index,
                                Vector2f position)
{
    // normal (circle iterator)
    int map_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (map_idx < xy_num)
    {
        // for col-major
        int map_idx_true = map_idx / size_pos[1] + (map_idx % size_pos[1]) * size_pos[0];
        if (!map_nan[map_idx_true] || inpaint_method!=0)
        {
            Vector2i idx(map_idx/size_pos[1], map_idx%size_pos[1]);
            Vector2f origin_off(origin[0], origin[1]);
            origin_off[0] -= 0.5 * resolution_pos;
            origin_off[1] -= 0.5 * resolution_pos;
            if ( start_index[0] != 0 || start_index[1] != 0)
            {
                idx[0] -= start_index[0];
                idx[1] -= start_index[1];
                for (int i=0; i<2; i++)
                    wrapIndexOne(idx[i], size_pos[i]);
            }
            Vector2f center = position + origin_off - (idx.cast<float>() * resolution_pos).matrix();
            Vector2f top_left = center.array() + normal_radius;
            Vector2f bottom_right = center.array() - normal_radius;
            boundPos(top_left, position);
            boundPos(bottom_right, position);
            Array2i sub_start;
            sub_start[0] = (int) ( (origin[0] + position[0] - top_left[0]) * resolution_pos_inv );
            sub_start[1] = (int) ( (origin[1] + position[1] - top_left[1]) * resolution_pos_inv );
            Array2i sub_end;
            sub_end[0] = (int) ( (origin[0] + position[0] - bottom_right[0]) * resolution_pos_inv );
            sub_end[1] = (int) ( (origin[1] + position[1] - bottom_right[1]) * resolution_pos_inv );
            if ( start_index[0] != 0 || start_index[1] != 0)
            {
                sub_start[0] += start_index[0];
                sub_start[1] += start_index[1];
                for (int i=0; i<2; i++)
                    wrapIndexOne(sub_start[i], size_pos[i]);
                sub_end[0] += start_index[0];
                sub_end[1] += start_index[1];
                for (int i=0; i<2; i++)
                    wrapIndexOne(sub_end[i], size_pos[i]);
                
                sub_start[0] -= start_index[0];
                sub_start[1] -= start_index[1];
                for (int i=0; i<2; i++)
                    wrapIndexOne(sub_start[i], size_pos[i]);
                sub_end[0] -= start_index[0];
                sub_end[1] -= start_index[1];
                for (int i=0; i<2; i++)
                    wrapIndexOne(sub_end[i], size_pos[i]);
            }
            Array2i buffer_size = sub_end - sub_start + Array2i::Ones();
            Vector3f mean_p = Vector3f::Zero();
            int cnt_p = 0;
            Vector3f temp_poses[60];
            for (int i=0; i<buffer_size[0]; i++)
            {
                for (int j=0; j<buffer_size[1]; j++)
                {
                    Array2i tempIndex = sub_start + Array2i(i, j);
                    if ( start_index[0] != 0 || start_index[1] != 0)
                    {
                        tempIndex += start_index;
                        for (int i = 0; i < tempIndex.size(); i++)
                            wrapIndexOne(tempIndex(i), size_pos[i]);
                    }
                    
                    // col-major
                    int true_idx = tempIndex[1] * size_pos[0] + tempIndex[0];
                    if (!map_nan[true_idx] || inpaint_method!=0)
                    {
                        if ( start_index[0] != 0 || start_index[1] != 0)
                        {
                            tempIndex[0] -= start_index[0];
                            tempIndex[1] -= start_index[1];
                            for (int i=0; i<2; i++)
                                wrapIndexOne(tempIndex[i], size_pos[i]);
                        }
                        Vector2f temp_pos = position + origin_off - (tempIndex.cast<float>() * resolution_pos).matrix();
                        if ((temp_pos - center).norm() < normal_radius)
                        {
                            Vector3f temp_pos3;
                            temp_pos3.head(2) = temp_pos;
                            temp_pos3[2] = inpainted[true_idx];
                            mean_p[0] += temp_pos[0];
                            mean_p[1] += temp_pos[1];
                            mean_p[2] += temp_pos3[2];
                            temp_poses[cnt_p] = temp_pos3;
                            cnt_p ++;
                        }
                    }
                }
            }

            mean_p[0] = mean_p[0] / (float)cnt_p;
            mean_p[1] = mean_p[1] / (float)cnt_p;
            mean_p[2] = mean_p[2] / (float)cnt_p;
            
            smooth[map_idx_true] = mean_p[2];
            if (cnt_p > 7)
            {
                float pMatrix[9] = {0};
                for(int i = 0; i < cnt_p; i ++)
                {
                    pMatrix[0] = pMatrix[0] + (temp_poses[i][0] - mean_p[0]) * (temp_poses[i][0] - mean_p[0]);
                    pMatrix[4] = pMatrix[4] + (temp_poses[i][1] - mean_p[1]) * (temp_poses[i][1] - mean_p[1]);
                    pMatrix[8] = pMatrix[8] + (temp_poses[i][2] - mean_p[2]) * (temp_poses[i][2] - mean_p[2]);
                    pMatrix[1] = pMatrix[1] + (temp_poses[i][0] - mean_p[0]) * (temp_poses[i][1] - mean_p[1]);
                    pMatrix[2] = pMatrix[2] + (temp_poses[i][0] - mean_p[0]) * (temp_poses[i][2] - mean_p[2]);
                    pMatrix[5] = pMatrix[5] + (temp_poses[i][1] - mean_p[1]) * (temp_poses[i][2] - mean_p[2]);
                    pMatrix[3] = pMatrix[1];
                    pMatrix[6] = pMatrix[2];
                    pMatrix[7] = pMatrix[5];
                }
                
                float dbEps = 0.01;
                int nJt = 30;
                float normal_vec[3];
                float curvature;
                computerEigenvalue(pMatrix, 3, normal_vec, &curvature, dbEps, nJt);
                
                float cos_xi;
                if (normal_vec[2] > 0)
                {
                    normal_x[map_idx_true] = normal_vec[0];
                    normal_y[map_idx_true] = normal_vec[1];
                    cos_xi = normal_vec[2];
                }
                else
                {
                    normal_x[map_idx_true] = -normal_vec[0];
                    normal_y[map_idx_true] = -normal_vec[1];
                    cos_xi = -normal_vec[2];
                }

                if (cos_xi < min_cosxi || curvature > max_curvature)
                {
                    risk[map_idx_true] = 1.0;
                    sdf[map_idx_true] = 1.0;
                }
                else
                {
                    risk[map_idx_true] = weight_xi * (1.0 - cos_xi) / (1.0 - min_cosxi) + weight_curvature * curvature;
                    sdf[map_idx_true] = 0.0;
                }
            }
            else
            {
                risk[map_idx_true] = 0.0;
                sdf[map_idx_true] = 0.0;
                normal_x[map_idx_true] = 0.0;
                normal_y[map_idx_true] = 0.0;
            }
        }

        // aggresive
        if (map_nan[map_idx_true])
        {
            risk[map_idx_true] = 0.0;
            sdf[map_idx_true] = 0.0;
        }
    }
}

__global__ void compute_sdf(Array2i start_index,
                            Vector2f position,
                            int dim)
{
    ;
}

void gpuInit(const Vector2i& size_pos_,
            int   size_yaw_,
            const Vector2f& length_pos_,
            float resolution_pos_,
            float resolution_yaw_,
            const Vector2f& position_,
            float min_var_,
            float max_var_,
            float hori_var_,
            float mahalanobis_threshold_,
            float normal_radius_,
            float min_cosxi_,
            float max_curvature_,
            float weight_xi_,
            float weight_curvature_,
            float ignore_z_min_,
            float ignore_z_max_,
            float beam_sigma2_,
            const Vector3f& body2sensor_T_)
{
    // set up
    cudaSetDevice(0);
    bool h_map_nan[GRID_NUM];
    float h_elevation[GRID_NUM];
    float h_var[GRID_NUM];
    float h_inpainted[GRID_NUM];
    float h_smooth[GRID_NUM];
    float h_normal_x[GRID_NUM];
    float h_normal_y[GRID_NUM];
    float h_risk[GRID_NUM];
    float h_sdf[GRID_NUM];

    int h_temp_idx[CLOUD_NUM];
    float h_temp_var[CLOUD_NUM];
    float h_temp_height[CLOUD_NUM];

    checkCudaError(cudaMemcpyToSymbol(map_nan, &h_map_nan, sizeof(bool)*GRID_NUM, 0, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyToSymbol(elevation, &h_elevation, sizeof(float)*GRID_NUM, 0, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyToSymbol(var, &h_var, sizeof(float)*GRID_NUM, 0, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyToSymbol(inpainted, &h_inpainted, sizeof(float)*GRID_NUM, 0, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyToSymbol(smooth, &h_smooth, sizeof(float)*GRID_NUM, 0, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyToSymbol(normal_x, &h_normal_x, sizeof(float)*GRID_NUM, 0, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyToSymbol(normal_y, &h_normal_y, sizeof(float)*GRID_NUM, 0, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyToSymbol(risk, &h_risk, sizeof(float)*GRID_NUM, 0, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyToSymbol(sdf, &h_sdf, sizeof(float)*GRID_NUM, 0, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyToSymbol(temp_idx, &h_temp_idx, sizeof(int)*CLOUD_NUM, 0, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyToSymbol(temp_var, &h_temp_var, sizeof(float)*CLOUD_NUM, 0, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyToSymbol(temp_height, &h_temp_height, sizeof(float)*CLOUD_NUM, 0, cudaMemcpyHostToDevice));
    
    // get params
    int xy_num_ = size_pos_[0] * size_pos_[1];
    int xyyaw_num_ = xy_num_ * size_yaw_;
    float pix2 = 6.284f;
    float res_pos_inv_ = 1.0f / resolution_pos_;
    float res_yaw_inv_ = 1.0f / resolution_yaw_;
    float origin_[2];
    origin_[0] = 0.5 * length_pos_[0];
    origin_[1] = 0.5 * length_pos_[1];
    cudaMemcpyToSymbol(size_pos, &size_pos_, sizeof(int) * 2);
    cudaMemcpyToSymbol(size_yaw, &size_yaw_, sizeof(int));
    cudaMemcpyToSymbol(xy_num, &xy_num_, sizeof(int));
    cudaMemcpyToSymbol(xyyaw_num, &xyyaw_num_, sizeof(int));
    cudaMemcpyToSymbol(resolution_pos_inv, &res_pos_inv_, sizeof(float));
    cudaMemcpyToSymbol(resolution_yaw_inv, &res_yaw_inv_, sizeof(float));
    cudaMemcpyToSymbol(resolution_pos, &resolution_pos_, sizeof(float));
    cudaMemcpyToSymbol(resolution_yaw, &resolution_yaw_, sizeof(float));
    cudaMemcpyToSymbol(length_pos, &length_pos_, sizeof(float) * 2);
    cudaMemcpyToSymbol(length_yaw, &pix2, sizeof(float));
    cudaMemcpyToSymbol(origin, &origin_, sizeof(float) * 2);
    cudaMemcpyToSymbol(min_var, &min_var_, sizeof(float));
    cudaMemcpyToSymbol(max_var, &max_var_, sizeof(float));
    cudaMemcpyToSymbol(hori_var, &hori_var_, sizeof(float));
    cudaMemcpyToSymbol(mahalanobis_threshold, &mahalanobis_threshold_, sizeof(float)); 
    cudaMemcpyToSymbol(normal_radius, &normal_radius_, sizeof(float));
    cudaMemcpyToSymbol(min_cosxi, &min_cosxi_, sizeof(float));
    cudaMemcpyToSymbol(max_curvature, &max_curvature_, sizeof(float));
    cudaMemcpyToSymbol(weight_xi, &weight_xi_, sizeof(float));
    cudaMemcpyToSymbol(weight_curvature, &weight_curvature_, sizeof(float));
    cudaMemcpyToSymbol(ignore_z_min, &ignore_z_min_, sizeof(float));
    cudaMemcpyToSymbol(ignore_z_max, &ignore_z_max_, sizeof(float));
    cudaMemcpyToSymbol(body2sensor_T, &body2sensor_T_, sizeof(float) * 3);
    cudaMemcpyToSymbol(beam_sigma2, &beam_sigma2_, sizeof(float));
    int inpaint_method_ = 1;
    cudaMemcpyToSymbol(inpaint_method, &inpaint_method_, sizeof(float));

    int num_block =(xy_num_ + THREAD_NUM - 1) / THREAD_NUM;
	init_map<<<num_block, THREAD_NUM>>>();
}

void computerMapR2(const int& point_num_,
                    float* points_world,
                    const float& resolution_pos_,
                    const Vector2i& size_pos_,
                    const Vector3f& body_T_,
                    const Matrix3f& body_R_,
                    const Matrix4f& body_Cov_ZR_,
                    MatrixXf& elevation_,
                    MatrixXf& var_,
                    MatrixXf& inpainted_,
                    MatrixXf& smooth_,
                    MatrixXf& normal_x_,
                    MatrixXf& normal_y_,
                    MatrixXf& risk_,
                    MatrixXf& sdf_,
                    Array2i& start_idx_,
                    Vector2f& position_,
                    bool verbose_)
{
    // move map
    Array2i indexShift;
    Vector2f positionShift = body_T_.head(2) - position_;
    
    Vector2f positionShiftTemp = (positionShift.array() / resolution_pos_).matrix();
    for (int i = 0; i < indexShift.size(); i++)
        indexShift[i] = static_cast<int>(positionShiftTemp[i] + 0.5 * (positionShiftTemp[i] > 0 ? 1 : -1));

    Vector2f alignedPositionShift = (indexShift.cast<float>() * resolution_pos_).matrix();
    indexShift = -indexShift;

    // Delete fields that fall out of map (and become empty cells).
    bool clear_all_ = false;
    bool clear_two_ = false;
    Vector2i clear_idx_ = Vector2i::Zero();
    Vector2i clear_nums_ = Vector2i::Zero();
    Vector2i clear_idx2_ = Vector2i::Zero();
    Vector2i clear_nums2_ = Vector2i::Zero();
    
    for (int i = 0; i < indexShift.size(); i++)
    {
        if (indexShift(i) != 0)
        {
            if (abs(indexShift(i)) >= size_pos_[i])
                clear_all_ = true;
            else
            {
                // Drop cells out of map.
                int sign = (indexShift(i) > 0 ? 1 : -1);
                int startIndex = start_idx_[i] - (sign < 0 ? 1 : 0);
                int endIndex = startIndex - sign + indexShift(i);
                int nCells = abs(indexShift(i));
                int index = (sign > 0 ? startIndex : endIndex);
                wrapIndexOne(index, size_pos_[i]);

                if (index + nCells <= size_pos_[i])
                {
                    // One region to drop.
                    clear_idx_[i] = index;
                    clear_nums_[i] = nCells;
                }
                else
                {
                    // Two regions to drop.
                    clear_two_ = true;
                    int firstIndex = index;
                    int firstNCells = size_pos_[i] - firstIndex;
                    clear_idx_[i] = firstIndex;
                    clear_nums_[i] = firstNCells;

                    int secondIndex = 0;
                    int secondNCells = nCells - firstNCells;
                    clear_idx2_[i] = secondIndex;
                    clear_nums2_[i] = secondNCells;
                }
            }
        }
    }
    
    // clear map
    int xy_num_ = size_pos_[0]*size_pos_[1];
    int num_block =(xy_num_ + THREAD_NUM - 1) / THREAD_NUM;
	clear_map<<<num_block, THREAD_NUM>>>(clear_idx_, 
                                        clear_nums_, 
                                        clear_idx2_,
                                        clear_nums2_,
                                        clear_all_,
                                        clear_two_);

    // update information.
    start_idx_[0] += indexShift[0];
    start_idx_[1] += indexShift[1];
    wrapIndexOne(start_idx_[0], size_pos_[0]);
    wrapIndexOne(start_idx_[1], size_pos_[1]);
    position_[0] += alignedPositionShift[0];
    position_[1] += alignedPositionShift[1];

    // get points
    Vector3f world_T_ = -body_R_.transpose() * body_T_;
    float* dev_points;
    cudaMalloc((void**)&dev_points, point_num_ * sizeof(float) * 3);
    cudaMemcpy(dev_points, points_world, point_num_ * sizeof(float) * 3, cudaMemcpyHostToDevice);

    // compute var
    int blocksPerGrid =(point_num_ + THREAD_NUM - 1) / THREAD_NUM;
	compute_var<<<blocksPerGrid, THREAD_NUM>>>(point_num_,
                                                dev_points, 
                                                world_T_,
                                                body_R_,
                                                body_Cov_ZR_.bottomRightCorner(3, 3),
                                                body_Cov_ZR_(0, 0),
                                                start_idx_,
                                                position_
                                                );
    // usleep(1);
    cudaDeviceSynchronize();
    cudaFree(dev_points);

    // compute map
    blocksPerGrid =(xy_num_ + THREAD_NUM - 1) / THREAD_NUM;
	compute_elevation<<<blocksPerGrid, THREAD_NUM>>>(point_num_);
    // usleep(1);
    cudaDeviceSynchronize();

    // inpaint map
    compute_inpaint_spiral<<<blocksPerGrid, THREAD_NUM>>>(start_idx_, position_);
    cudaDeviceSynchronize();

    // compute r2 normal and cost
    compute_map_r2<<<blocksPerGrid, THREAD_NUM>>>(start_idx_, position_);
    cudaDeviceSynchronize();

    // get map
    if (verbose_)
    {
        cudaMemcpyFromSymbol(elevation_.data(), elevation, sizeof(float)*xy_num_);
        cudaMemcpyFromSymbol(var_.data(), var, sizeof(float)*xy_num_);
        cudaMemcpyFromSymbol(inpainted_.data(), inpainted, sizeof(float)*xy_num_);
        cudaMemcpyFromSymbol(smooth_.data(), smooth, sizeof(float)*xy_num_);
        cudaMemcpyFromSymbol(normal_x_.data(), normal_x, sizeof(float)*xy_num_);
        cudaMemcpyFromSymbol(normal_y_.data(), normal_y, sizeof(float)*xy_num_);
        cudaMemcpyFromSymbol(risk_.data(), risk, sizeof(float)*xy_num_);
    }
    cudaMemcpyFromSymbol(sdf_.data(), sdf, sizeof(float)*xy_num_);
    cudaDeviceSynchronize();
}
