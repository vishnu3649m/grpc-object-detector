/**
 * A detector that returns random pokemons.
 * Created for experimenting and learning the intricacies of gRPC server operation.
 */

#ifndef GRPC_OBJ_DET_RANDOMPOKEMONDETECTOR_H
#define GRPC_OBJ_DET_RANDOMPOKEMONDETECTOR_H

#include <absl/random/random.h>

#include "DetectorInterface.h"

static ObjDet::RectTLWH generate_random_box() {
  absl::BitGen bitgen;
  float xmin = absl::Uniform<float>(bitgen, 0.05, 0.75);
  float ymin = absl::Uniform<float>(bitgen, 0.05, 0.6);
  float xmax = xmin + absl::Uniform<float>(bitgen, 0.15, 0.25);
  float ymax = ymin +  absl::Uniform<float>(bitgen, 0.2, 0.4);

  return ObjDet::RectTLWH(xmin, ymin, xmax, ymax);
}

namespace ObjDet {
  class RandomPokemonDetector final : public DetectorInterface {
    std::string name = "random_pokemon";
    std::string model = "random";
    std::vector<std::string> class_label_map {"charmander",
                                              "pikachu",
                                              "squirtle",
                                              "eevee",
                                              "lucario",
                                              "magikarp",
                                              "mewtwo"};
    bool init = false;

  public:
    RandomPokemonDetector() = default;
    ~RandomPokemonDetector() override = default;

    void initialize() override {
      init  = true;
    }

    std::vector<Detection>  detect(const cv::Mat &img) override {
      absl::BitGen bitgen;
      int n_detections = absl::Uniform<int>(bitgen, 1, 10);
      std::vector<Detection> output;

      for (int i = 0; i < n_detections; i++) {
        Detection det{
          absl::Uniform<int>(bitgen, 0u, class_label_map.size()),
          generate_random_box(),
          absl::Uniform<float>(bitgen, 0.4, 0.99)
        };
        output.push_back(det);
      }

      return output;
    }

    std::pair<std::string, std::string> describe() const override {
      return {name, model};
    }

    std::unordered_set<std::string> available_objects_lookup() const override  {
      return {class_label_map.begin(), class_label_map.end()};
    }

    std::string class_id_to_label(int class_id) const override {
      return class_label_map[class_id];
    }

    bool is_initialized() const override {
      return init;
    }
  };
}

#endif //GRPC_OBJ_DET_RANDOMPOKEMONDETECTOR_H
