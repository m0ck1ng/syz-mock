typedef struct model_t model_t;
model_t* model_default();
model_t* model_new(const char *path, uint32_t device_id);
void model_load(model_t *m, const char *path, uint32_t device_id);
int32_t model_mutate(model_t *m, uint32_t *numbers, uint32_t length, uint32_t idx);
int32_t model_gen(model_t *m, uint32_t *numbers, uint32_t length);
uint32_t model_exists(model_t *m);