#ifdef BUILD_NAMEDTENSOR
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <c10/util/Exception.h>
#include <torch/csrc/utils/memory.h>

using at::Dimname;
using at::DimnameList;
using at::NamedTensorMeta;
using at::Symbol;
using torch::make_unique;

TEST(NamedTensorTest, defaultMetadata) {
  int num_names = 4;
  const auto meta = NamedTensorMeta(num_names);
  for (const auto name : meta.names()) {
    ASSERT_EQ(name.type(), at::NameType::WILDCARD);
  }
}

static Dimname dimnameFromString(const std::string& str) {
  return Dimname::fromSymbol(Symbol::dimname(str));
}

TEST(NamedTensorTest, isNamed) {
  auto tensor = at::zeros({3, 2, 5, 7});
  ASSERT_FALSE(tensor.has_names());

  tensor = at::zeros({3, 2, 5, 7});
  tensor.unsafeGetTensorImpl()->set_named_tensor_meta(
      make_unique<NamedTensorMeta>(tensor.dim()));
  ASSERT_FALSE(tensor.has_names());

  tensor = at::zeros({3, 2, 5, 7});
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  std::vector<Dimname> names = { N, C, H, W };
  tensor.unsafeGetTensorImpl()->set_named_tensor_meta(
      make_unique<NamedTensorMeta>(names));
  ASSERT_TRUE(tensor.has_names());
}

static bool dimnames_equal(at::DimnameList names, at::DimnameList other) {
  if (names.size() != other.size()) {
    return false;
  }
  for (auto i = 0; i < names.size(); i++) {
    const auto& name = names[i];
    const auto& other_name = other[i];
    if (name.type() != other_name.type() || name.full_name() != other_name.full_name()) {
      return false;
    }
  }
  return true;
}

TEST(NamedTensorTest, attachMetadata) {
  auto tensor = at::zeros({3, 2, 5, 7});
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  std::vector<Dimname> names = { N, C, H, W };

  tensor.unsafeGetTensorImpl()->set_named_tensor_meta(
      make_unique<NamedTensorMeta>(names));
  
  const auto retrieved_meta = tensor.get_named_tensor_meta();
  ASSERT_TRUE(dimnames_equal(retrieved_meta->names(), names));

  // Test dropping metadata
  tensor.unsafeGetTensorImpl()->set_named_tensor_meta(nullptr);
  ASSERT_FALSE(tensor.has_names());
}

TEST(NamedTensorTest, internalSetNamesInplace) {
  auto tensor = at::zeros({3, 2, 5, 7});
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  std::vector<Dimname> names = { N, C, H, W };
  ASSERT_FALSE(tensor.has_names());

  // Set names
  at::internal_set_names_inplace(tensor, names);
  const auto retrieved_names = tensor.names().value();
  ASSERT_TRUE(dimnames_equal(retrieved_names, names));

  // Drop names
  at::internal_set_names_inplace(tensor, at::nullopt);
  ASSERT_TRUE(tensor.get_named_tensor_meta() == nullptr);
  ASSERT_TRUE(tensor.names() == at::nullopt);
}

TEST(NamedTensorTest, empty) {
  auto N = Dimname::fromSymbol(Symbol::dimname("N"));
  auto C = Dimname::fromSymbol(Symbol::dimname("C"));
  auto H = Dimname::fromSymbol(Symbol::dimname("H"));
  auto W = Dimname::fromSymbol(Symbol::dimname("W"));
  std::vector<Dimname> names = { N, C, H, W };

  auto tensor = at::empty({});
  ASSERT_EQ(tensor.names(), at::nullopt);

  tensor = at::empty({1, 2, 3});
  ASSERT_EQ(tensor.names(), at::nullopt);

  tensor = at::empty({1, 2, 3, 4}, names);
  ASSERT_TRUE(dimnames_equal(tensor.names().value(), names));

  ASSERT_THROW(at::empty({1, 2, 3}, names), c10::Error);
}

TEST(NamedTensorTest, dimnameToPosition) {
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  std::vector<Dimname> names = { N, C, H, W };

  auto tensor = at::empty({1, 1, 1});
  ASSERT_THROW(dimname_to_position(tensor, N), c10::Error);

  tensor = at::empty({1, 1, 1, 1}, names);
  ASSERT_EQ(dimname_to_position(tensor, H), 2);

  auto Cin = dimnameFromString("C.in");
  auto Cout = dimnameFromString("C.out");
  tensor = at::empty({1, 1, 1, 1}, names);
  ASSERT_THROW(dimname_to_position(tensor, Cin), c10::Error);

  tensor = at::empty({1, 1}, std::vector<Dimname>({ Cin, Cout }));
  ASSERT_THROW(dimname_to_position(tensor, C), c10::Error);

  tensor = at::empty({1, 1}, std::vector<Dimname>({ Cin, N }));
  ASSERT_EQ(dimname_to_position(tensor, C), 0);
}

static void check_unify(
    DimnameList names,
    DimnameList other_names,
    DimnameList expected) {
  const auto result = at::unify_from_right(names, other_names);
  ASSERT_TRUE(dimnames_equal(result.value(), expected));
}

static void check_unify_error(DimnameList names, DimnameList other_names) {
  ASSERT_THROW(at::unify_from_right(names, other_names), c10::Error);
}

TEST(NamedTensorTest, unifyFromRight) {
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  auto None = dimnameFromString("*");
  
  std::vector<Dimname> names = { N, C };
  ASSERT_TRUE(dimnames_equal(*at::unify_from_right(at::nullopt, names), names));
  ASSERT_TRUE(dimnames_equal(*at::unify_from_right(names, at::nullopt), names));
  ASSERT_FALSE(at::unify_from_right(at::nullopt, at::nullopt).has_value());

  check_unify({ N, C, H, W }, { N, C, H, W }, { N, C, H, W });
  check_unify({ W }, { C, H, W }, { C, H, W });
  check_unify({ None, W }, { C, H, W }, { C, H, W });
  check_unify({ None, None, H, None }, { C, None, W }, { None, C, H, W });

  check_unify_error({ W, H }, { W, C });
  check_unify_error({ W, H }, { C, H });
  check_unify_error({ None, H }, { H, None });
  check_unify_error({ H, None, C }, { H });
}

#endif
