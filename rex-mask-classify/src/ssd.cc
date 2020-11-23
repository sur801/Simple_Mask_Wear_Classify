#include "ssd.h"

//char EMOTIONS[NUM_CLASS][10] = { "angry", "disgust", "fearful", "happy", "sad", "surprised", "neutral" }; // 7가지 표정

char MASK_STATE[2][20] = {"WIHTOUT MASK", "WITH MASK"};

void ssd_post(float* outputs, int w, int h, ssd_ctx* ctx)
{
	int i;
	float *results = outputs; // rknn output

	int max_label = results[0] > results[1] ? 0 : 1;
	printf(">> res0 : %f , res1 : %f \n", results[0], results[1]);
	printf(">> %s\n", MASK_STATE[max_label]); // 더 높은 값이 나온 label 출력
	//0번째는 마스크 쓴 상태, 1번째는 마스크 안쓴 상태.
}

char *readLine(FILE *fp, char *buffer, int *len) 
{
    int ch;
    int i = 0;
    size_t buff_len = 0;

    buffer = (char *)malloc(buff_len + 1);
    if (!buffer)
        return NULL; // Out of memory

    while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL) {
            free(buffer);
            return NULL; // Out of memory
        }
        buffer = (char *)tmp;

        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';

    *len = buff_len;

    // Detect end
    if (ch == EOF && (i == 0 || ferror(fp))) {
        free(buffer);
        return NULL;
    }
    return buffer;
}

int readLines(const char *fileName, char *lines[]) 
{
    FILE *file = fopen(fileName, "r");
    char *s;
    int i = 0;
    int n = 0;
    while ((s = readLine(file, s, &n)) != NULL) {
        lines[i++] = s;
    }
    return i;
}

int loadLabelName(const char *locationFilename, char *labels[]) 
{
    readLines(locationFilename, labels);
    return 0;
}




int ssd_init(const char *model_name, const char *label_path, ssd_ctx *ctx)
{
	int ret;

	ret = rknn_init_helper(model_name, &ctx->rknn);
	if (ret != 0) {
		fprintf(stderr, "%s : Failed to load model", __func__);
		return -1;
	}

	ret = loadLabelName(label_path, ctx->labels);
	if (ret != 0) {
		fprintf(stderr, "%s : Failed to load label", __func__);
		return -1;
	}

	return 0;
}

int ssd_run(ssd_ctx *ctx, uint8_t *img, int w, int h, ssize_t size)
{
	int ret;
	rknn_input inputs[1];
	rknn_output outputs[1];

	memset(inputs, 0x00, sizeof(inputs));
	inputs[0].index = 0;
	inputs[0].type = RKNN_TENSOR_UINT8;
	inputs[0].size = size;
	inputs[0].fmt = RKNN_TENSOR_NHWC;
	inputs[0].buf = img;

	//printf("input set start \n");
	ret = rknn_inputs_set(ctx->rknn, 1, inputs);
	if (ret < 0) {
		fprintf(stdout, "%s fail\n", __func__);
		return -1;
	}
	//printf("input set \n");
	ret = rknn_run(ctx->rknn, NULL);
	if (ret < 0) {
		fprintf(stdout, "%s fail\n", __func__);
		return -1;
	}

	//printf("rknn_run done\n");

	memset(outputs, 0x00, sizeof(outputs));
	outputs[0].want_float = 1;

	ret = rknn_outputs_get(ctx->rknn, 1, outputs, NULL);
	if (ret < 0) {
		fprintf(stdout, "%s fail\n", __func__);
		return -1;
	}

	//printf("get output done\n");
	//SSD 후처리코드
	ssd_post((float *)outputs[0].buf, w, h, ctx);
	rknn_outputs_release(ctx->rknn, 1, outputs);

	return 0;
}
