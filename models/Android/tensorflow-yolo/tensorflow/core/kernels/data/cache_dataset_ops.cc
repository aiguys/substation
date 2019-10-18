/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/kernels/data/cache_dataset_ops.h"

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/cache_ops.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level description of
// the following op.

/* static */ constexpr const char* const CacheDatasetOp::kDatasetType;
/* static */ constexpr const char* const CacheDatasetOp::kInputDataset;
/* static */ constexpr const char* const CacheDatasetOp::kFileName;
/* static */ constexpr const char* const CacheDatasetOp::kOutputTypes;
/* static */ constexpr const char* const CacheDatasetOp::kOutputShapes;

constexpr char kKeyStrFormat[] = "%%%zuzu_%%%zuzu";
constexpr char kPaddingSizeStrFormat[] = "%zu";
constexpr char kFileDatasetPrefix[] = "File";
constexpr char kMode[] = "Mode";
constexpr char kLockFileSuffix[] = ".lockfile";
constexpr char kIterationCompleted[] = "iteration_completed";
constexpr char kCurIndex[] = "cur_index";
constexpr char kShardId[] = "shard_id";
constexpr char kCreatedAt[] = "Created at";
constexpr char kMemoryDatasetPrefix[] = "Memory";
constexpr char kMemoryCache[] = "MemoryCache";
constexpr char kTFData[] = "tf_data";
constexpr char kCacheClaimed[] = "cache_claimed";
constexpr char kCacheSize[] = "cache_size";
constexpr char kCache[] = "cache";
constexpr char kSizeSuffix[] = ".size";
constexpr char kCacheCompleted[] = "cache_completed";
constexpr char kIndex[] = "index";
constexpr char kImpl[] = "Impl";
constexpr char kCacheDataset[] = "CacheDataset";

class CacheDatasetOp::FileDataset : public DatasetBase {
 public:
  explicit FileDataset(OpKernelContext* ctx, const DatasetBase* input,
                       string filename, Env* env)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        filename_(std::move(filename)),
        env_(env),
        num_tensors_(input->output_dtypes().size()),
        tensor_index_padding_size_(StringPaddingSize(num_tensors_)),
        item_index_padding_size_(StringPaddingSize(kMaxItems)),
        tensor_format_string_(strings::Printf(kKeyStrFormat,
                                              item_index_padding_size_,
                                              tensor_index_padding_size_)) {
    input_->Ref();
    DCHECK_EQ(item_index_padding_size_, 7);
  }

  ~FileDataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    name_utils::IteratorPrefixParams params;
    params.dataset_prefix = kFileDatasetPrefix;
    return absl::make_unique<FileIterator>(FileIterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix, params)});
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.dataset_prefix = kFileDatasetPrefix;
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  int64 Cardinality() const override { return input_->Cardinality(); }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph));
    Node* filename = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(filename_, &filename));
    TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph, filename}, output));
    return Status::OK();
  }

  const DatasetBase* const input_;
  const tstring filename_;

 private:
  static size_t StringPaddingSize(size_t num_tensors) {
    return strings::Printf(kPaddingSizeStrFormat, num_tensors - 1).size();
  }

  string FormatName(size_t item_index, size_t tensor_index) const {
    return strings::Printf(tensor_format_string_.c_str(), item_index,
                           tensor_index);
  }

  class FileIterator : public DatasetIterator<FileDataset> {
   public:
    explicit FileIterator(const Params& params)
        : DatasetIterator<FileDataset>(params) {
      if (params.dataset->env_
              ->FileExists(MetaFilename(params.dataset->filename_))
              .ok()) {
        mode_ = Mode::read;
      } else {
        mode_ = Mode::write;
      }
      InitializeIterator();
    }

    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(mu_);
      return iterator_->Initialize(ctx);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      return iterator_->GetNext(ctx, out_tensors, end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/1);
    }

    Status SaveInternal(IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kMode), mode_));
      return SaveInput(writer, iterator_);
    }
    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      {
        int64 temp;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kMode), &temp));
        mode_ = static_cast<Mode>(temp);
      }
      if (mode_ == Mode::write &&
          dataset()
              ->env_->FileExists(MetaFilename(dataset()->filename_))
              .ok()) {
        // This could happen if the cache was completely written after the
        // checkpoint was saved.
        LOG(WARNING)
            << "It looks like the cache was already completely written("
            << MetaFilename(dataset()->filename_)
            << ") after the last checkpoint was saved. Attempting to read "
            << "the cache instead of continuing to write. If this is a "
            << "mistake, please remove the above file and try running again.";
        mode_ = Mode::read;
      }
      InitializeIterator();
      TF_RETURN_IF_ERROR(iterator_->Initialize(ctx));
      return RestoreInput(ctx, reader, iterator_);
    }

   private:
    // FileWriterIterator passes through and caches items from the input
    // FileDataset.
    //
    // This iterator is used when the cache directory is not found on disk. It
    // creates the cache directory, and passes on the underlying iterator's
    // elements.
    //
    // Caching is performed by writing the input tensors to disk using the
    // `BundleWriter`. Note that the cache gets fully flushed to disk only
    // after the input iterator has been fully exhausted. If the program
    // exits, before completion of an epoch, the cached state would be lost.
    // To ensure that the partial cache persists across sessions, one should
    // checkpoint the input pipeline. On each call to `SaveInternal` the
    // partial cache gets flushed to disk in files with prefix
    // <filename>_<shard_id> where shard_id is unique for each checkpoint.
    // When all elements have been produced, these shards get coalesced.
    class FileWriterIterator : public DatasetIterator<FileDataset> {
     public:
      explicit FileWriterIterator(const Params& params)
          : DatasetIterator<FileDataset>(params),
            cur_index_(0),
            shard_id_(0),
            filename_(
                strings::StrCat(params.dataset->filename_, "_", shard_id_)),
            lockfile_(strings::StrCat(filename_, kLockFileSuffix)),
            lockfile_created_(false),
            iteration_completed_(false) {}

      ~FileWriterIterator() {
        if (!dataset()->env_->FileExists(MetaFilename(filename_)).ok()) {
          std::vector<string> cache_files;
          Status s = dataset()->env_->GetMatchingPaths(
              strings::StrCat(filename_, "*"), &cache_files);
          if (!s.ok()) {
            LOG(WARNING) << "Failed to get matching files on " << filename_
                         << "* : " << s.ToString();
          }
          for (const string& path : cache_files) {
            s = dataset()->env_->DeleteFile(path);
            if (!s.ok()) {
              LOG(WARNING) << "Failed to delete " << path << " : "
                           << s.ToString();
            }
          }
        }
      }

      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        *end_of_sequence = false;
        TF_RETURN_IF_ERROR(EnsureLockFileExists(end_of_sequence));
        if (*end_of_sequence) {
          return Status::OK();
        }
        TF_RETURN_IF_ERROR(writer_->status());
        if (cur_index_ >= kMaxItems) {
          // As a courtesy, close the [truncated] cache file.
          Status s = Finish();
          if (!s.ok()) {
            LOG(ERROR) << s;
          }
          return errors::InvalidArgument(
              "Upstream iterator is producing more than ", kMaxItems,
              " items, which is more than the cache limit.");
        }

        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
        if (*end_of_sequence && out_tensors->empty()) {
          TF_RETURN_IF_ERROR(Finish());
          cur_index_++;
          return Status::OK();
        }
        if (out_tensors->size() != dataset()->num_tensors_) {
          return errors::Internal(
              "Upstream iterator returned invalid number of tensors. "
              "Expected ",
              dataset()->num_tensors_, " got: ", out_tensors->size());
        }
        size_t tensor_index = 0;
        for (const Tensor& t : *out_tensors) {
          DCHECK_LT(tensor_index, dataset()->num_tensors_);
          string key = dataset()->FormatName(cur_index_, tensor_index++);
          TF_RETURN_IF_ERROR(writer_->Add(key, t));
        }
        if (*end_of_sequence) {
          TF_RETURN_IF_ERROR(Finish());
        }
        cur_index_++;
        return Status::OK();
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kCurIndex), cur_index_));

        if (iteration_completed_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name(kIterationCompleted), ""));
          return Status::OK();
        }

        // lockfile is created on the first call to GetNextInternal. The
        // absence of a lockfile means that GetNextInternal was not called
        // and hence nothing was written to cache. So we don't need to worry
        // about flushing the current shard. This ensures that we never write
        // empty shards.
        if (lockfile_created_) {
          // Flush the current bundle.
          TF_RETURN_IF_ERROR(writer_->Finish());

          // Note: We do not delete the lockfile here. We keep lockfiles of
          // all shards around until the entire cache has been written to
          // prevent concurrent iterators from corrupting any of the shards.

          // Start caching to a new shard.
          shard_id_++;
          filename_ = strings::StrCat(dataset()->filename_, "_", shard_id_);
          lockfile_ = strings::StrCat(filename_, kLockFileSuffix);
          lockfile_created_ = false;
        }
        TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kShardId), shard_id_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        int64 temp;
        // TODO(b/78048575): Update this when saving size_t tensors directly
        // is supported.
        {
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kCurIndex), &temp));
          cur_index_ = static_cast<size_t>(temp);
          if (cur_index_ != temp) {
            return errors::Internal("Invalid value for cur_index ", temp);
          }
        }

        if (reader->Contains(full_name(kIterationCompleted))) {
          iteration_completed_ = true;
          return Status::OK();
        }

        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));

        // TODO(b/78048575): Update this when saving size_t tensors directly
        // is supported.
        {
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kShardId), &temp));
          shard_id_ = static_cast<size_t>(temp);
          if (shard_id_ != temp) {
            return errors::Internal("Invalid value for shard_id ", temp);
          }
        }
        filename_ = strings::StrCat(dataset()->filename_, "_", shard_id_);
        lockfile_ = strings::StrCat(filename_, kLockFileSuffix);
        writer_ = absl::make_unique<BundleWriter>(dataset()->env_, filename_);
        return Status::OK();
      }

     private:
      Status EnsureLockFileExists(bool* end_of_sequence)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (iteration_completed_) {
          *end_of_sequence = true;
          return Status::OK();
        }
        if (lockfile_created_) {
          return Status::OK();
        }

        // Perform rudimentary locking to help catch concurrent writes to the
        // same cache files.

        // 1. Check that a checkpoint for the shard has not already been
        // written.
        if (dataset()->env_->FileExists(MetaFilename(filename_)).ok()) {
          return errors::AlreadyExists("Existing cache files found: \n",
                                       MetaFilename(filename_), "\n",
                                       DataFilename(filename_, 0, 1), "\n",
                                       "To continue delete the above files.");
        }

        // 2. Check that there isn't a concurrent iterator that is writing
        // to cache.
        if (dataset()->env_->FileExists(lockfile_).ok()) {
          // Attempt to read the contents of the lockfile.
          char contents_scratch[151] = {0};  // Initialize all to 0.
          StringPiece contents;
          std::unique_ptr<RandomAccessFile> file;
          if (dataset()->env_->NewRandomAccessFile(lockfile_, &file).ok()) {
            file->Read(0, 150, &contents, contents_scratch).IgnoreError();
          }
          return errors::AlreadyExists(
              "There appears to be a concurrent caching iterator running - "
              "cache lockfile already exists ('",
              lockfile_,
              "'). If you are sure no other running TF computations are "
              "using this cache prefix, delete the lockfile and "
              "re-initialize the iterator. Lockfile contents: ",
              contents);
        }
        // Create the file, and write some basic contents.
        std::unique_ptr<WritableFile> lockfile;
        TF_RETURN_IF_ERROR(
            dataset()->env_->NewWritableFile(lockfile_, &lockfile));
        TF_RETURN_IF_ERROR(lockfile->Append(
            strings::StrCat(kCreatedAt, ": ", dataset()->env_->NowSeconds())));

        // At this point we know that
        // 1. There is no conflicting checkpoint with prefix `filename_`.
        // 2. There is no concurrent session that is trying to write a ckpt
        //    to filename.
        // So it is safe to create a BundleWriter here. Note that it is
        // unsafe to initialize the BundleWriter anywhere the above
        // conditions are not met since BundleWriter's constructor creates
        // new temp files which can delete the temp files created by a
        // BundleWriter in another Session.
        writer_ = absl::make_unique<BundleWriter>(dataset()->env_, filename_);
        lockfile_created_ = true;
        return Status::OK();
      }

      Status Finish() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        iteration_completed_ = true;
        // Flush the current bundle.
        TF_RETURN_IF_ERROR(writer_->Finish());
        // Merge all the bundles.
        // Currently there are `shard_id_ + 1` bundles, one for each
        // checkpoint. Each bundle has prefix <filename>_<id> where `id` is an
        // integer starting at 0 and incremented by 1 for each new checkpoint.
        // We merge all these bundles into a bundle with prefix <filename> so
        // that the next call to `MakeIterator` can build a
        // `FileReaderIterator`.
        {
          std::vector<tstring> prefixes;
          prefixes.reserve(shard_id_ + 1);
          for (size_t i = 0; i <= shard_id_; ++i) {
            prefixes.emplace_back(
                strings::StrCat(dataset()->filename_, "_", i));
          }
          TF_RETURN_IF_ERROR(
              MergeBundles(dataset()->env_, prefixes, dataset()->filename_));
        }
        // Delete all lockfiles.
        for (size_t i = 0; i <= shard_id_; ++i) {
          TF_RETURN_IF_ERROR(dataset()->env_->DeleteFile(
              strings::StrCat(dataset()->filename_, "_", i, kLockFileSuffix)));
        }
        return Status::OK();
      }

      mutex mu_;
      size_t cur_index_ GUARDED_BY(mu_);
      // Index of the current shard. This gets incremented whenever a new
      // cache shard is saved.
      size_t shard_id_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      // The current prefix for the cache file. This is equal to
      // `StrCat(dataset()->filename_, "_", shard_id_)`.
      string filename_;
      std::unique_ptr<BundleWriter> writer_ GUARDED_BY(mu_);
      string lockfile_ GUARDED_BY(mu_);
      bool lockfile_created_ GUARDED_BY(mu_);
      bool iteration_completed_ GUARDED_BY(mu_);
    };  // FileWriterIterator

    class FileReaderIterator : public DatasetIterator<FileDataset> {
     public:
      explicit FileReaderIterator(const Params& params)
          : DatasetIterator<FileDataset>(params),
            cur_index_(0),
            reader_(dataset()->env_, dataset()->filename_),
            iterator_restored_(false) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        *end_of_sequence = false;
        TF_RETURN_IF_ERROR(reader_.status());
        if (!reader_.Valid()) {
          *end_of_sequence = true;
          return Status::OK();
        }
        out_tensors->clear();
        out_tensors->resize(dataset()->num_tensors_);

        for (size_t i = 0; i < dataset()->num_tensors_; ++i) {
          // When the iterator is restored from the checkpoint, `reader_` is
          // already pointing at `key` so we do not need to skip the header
          // entry.
          if (!iterator_restored_) {
            reader_.Next();  // The first entry in the table is a header.
          } else {
            iterator_restored_ = false;
          }
          if (!reader_.Valid()) {
            out_tensors->clear();
            *end_of_sequence = true;
            return Status::OK();
          }
          StringPiece key = reader_.key();
          DCHECK_EQ(key, dataset()->FormatName(cur_index_, i));
          TF_RETURN_IF_ERROR(reader_.ReadCurrent(&(*out_tensors)[i]));
          TF_RETURN_IF_ERROR(reader_.status());
        }
        cur_index_++;
        return Status::OK();
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kCurIndex), cur_index_));
        return Status::OK();
      }

      Status RestoreInternal(
          IteratorContext* ctx,
          IteratorStateReader* iterator_state_reader) override {
        mutex_lock l(mu_);
        {
          // TODO(b/78048575): Update this when saving size_t tensors directly
          // is supported.
          int64 temp;
          TF_RETURN_IF_ERROR(
              iterator_state_reader->ReadScalar(full_name(kCurIndex), &temp));
          cur_index_ = static_cast<size_t>(temp);
          if (cur_index_ != temp) {
            return errors::Internal("Invalid value for cur_index ", temp);
          }
        }
        if (!reader_.Valid()) {
          return errors::Internal("Error initializing BundleReader.");
        }
        reader_.Seek(dataset()->FormatName(cur_index_, 0));
        iterator_restored_ = true;
        return Status::OK();
      }

     private:
      mutex mu_;
      size_t cur_index_ GUARDED_BY(mu_);
      BundleReader reader_ GUARDED_BY(mu_);
      bool iterator_restored_ GUARDED_BY(mu_);
    };  // FileReaderIterator

    void InitializeIterator() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      // We intentionally use the same prefix for both `FileReaderIterator`
      // and `FileWriterIterator`. Since at any time there will be at most
      // one of them alive, there should be no conflicts. This allows both
      // iterators to use a common key for `cur_index`. We leverage this
      // in the corner case when this iterator is restored from an old
      // checkpoint in `write` mode and the cache has been completely
      // flushed to disk since then. In that case we simply build a
      // `FileReaderIterator` and seek to the `cur_index`.
      switch (mode_) {
        case Mode::read:
          iterator_ =
              absl::make_unique<FileReaderIterator>(FileReaderIterator::Params{
                  dataset(), strings::StrCat(prefix(), kImpl)});
          break;
        case Mode::write:
          iterator_ =
              absl::make_unique<FileWriterIterator>(FileWriterIterator::Params{
                  dataset(), strings::StrCat(prefix(), kImpl)});
      }
    }

    mutex mu_;
    enum Mode { read, write };
    Mode mode_ GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> iterator_ GUARDED_BY(mu_);
  };  // FileIterator

  Env* const env_;
  const size_t num_tensors_;
  const size_t tensor_index_padding_size_;
  static const size_t kMaxItems = 10000000;  // 10 million
  const size_t item_index_padding_size_;
  const string tensor_format_string_;
};  // FileDataset

class CacheDatasetOp::FileDatasetV2 : public CacheDatasetOp::FileDataset {
 public:
  explicit FileDatasetV2(OpKernelContext* ctx, const DatasetBase* input,
                         string filename, Env* env,
                         const Tensor& resource_handle)
      : FileDataset(ctx, input, filename, env),
        resource_handle_(resource_handle) {}

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));
    Node* filename_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(filename_, &filename_node));
    Node* resource_handle_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddTensor(resource_handle_, &resource_handle_node));
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {input_node, filename_node, resource_handle_node}, output));
    return Status::OK();
  }

 private:
  const Tensor resource_handle_;
};

class CacheDatasetOp::MemoryDataset : public DatasetBase {
 public:
  explicit MemoryDataset(OpKernelContext* ctx, const DatasetBase* input,
                         MemoryCache* cache)
      : DatasetBase(DatasetContext(ctx)), input_(input), cache_(cache) {
    input_->Ref();
  }

  ~MemoryDataset() override {
    input_->Unref();
    if (cache_) {
      cache_->Unref();
    }
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    name_utils::IteratorPrefixParams params;
    params.dataset_prefix = kMemoryDatasetPrefix;
    return absl::make_unique<MemoryIterator>(
        MemoryIterator::Params{
            this, name_utils::IteratorPrefix(kDatasetType, prefix, params)},
        cache_);
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.dataset_prefix = kMemoryDatasetPrefix;
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  int64 Cardinality() const override { return input_->Cardinality(); }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));
    Node* filename_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(tstring(""), &filename_node));
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_node, filename_node}, output));
    return Status::OK();
  }

  class MemoryIterator : public DatasetIterator<MemoryDataset> {
   public:
    explicit MemoryIterator(const Params& params, MemoryCache* cache)
        : DatasetIterator<MemoryDataset>(params), cache_(cache) {}

    ~MemoryIterator() override {
      if (dataset()->cache_ == nullptr) {
        cache_->Unref();
      }
    }

    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(mu_);
      if (cache_ == nullptr) {
        // Use the resource manager in the iterator context to get / create
        // a cache.
        ResourceMgr* mgr = ctx->resource_mgr();
        const string name = strings::StrCat(
            prefix(), name_utils::kDelimiter, dataset()->node_name(),
            name_utils::kDelimiter, kMemoryCache);
        TF_RETURN_IF_ERROR(mgr->LookupOrCreate<MemoryCache>(
            kTFData, name, &cache_, [](MemoryCache** cache) {
              *cache = new MemoryCache();
              return Status::OK();
            }));
      }
      mode_ = cache_->MaybeClaim() ? Mode::write : Mode::read;
      InitializeIterator();
      if (mode_ == Mode::read && !cache_->IsCompleted()) {
        return errors::Internal(
            "Cache should only be read after it has been completed.");
      }
      return iterator_->Initialize(ctx);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      return iterator_->GetNext(ctx, out_tensors, end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/1);
    }

    Status SaveInternal(IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kMode), mode_));
      if (cache_->IsClaimed()) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kCacheClaimed), ""));
        size_t cache_size = cache_->size();
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kCacheSize), cache_size));
        for (size_t i = 0; i < cache_size; i++) {
          auto& element = cache_->at(i);
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(kCache, "[", i, "]", kSizeSuffix)),
              element.size()));
          for (size_t j = 0; j < element.size(); ++j) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(strings::StrCat(kCache, "[", i, "][", j, "]")),
                element[j]));
          }
        }
        if (cache_->IsCompleted()) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name(kCacheCompleted), ""));
        }
      }
      return SaveInput(writer, iterator_);
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      iterator_.reset();
      cache_->Reset();
      {
        int64 temp;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kMode), &temp));
        mode_ = static_cast<Mode>(temp);
      }
      if (reader->Contains(full_name(kCacheClaimed))) {
        CHECK(cache_->MaybeClaim());
        size_t cache_size;
        {
          int64 temp;
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kCacheSize), &temp));
          cache_size = static_cast<size_t>(temp);
        }
        for (size_t i = 0; i < cache_size; ++i) {
          std::vector<Tensor> element;
          size_t element_size;
          {
            int64 temp;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat(kCache, "[", i, "]", kSizeSuffix)),
                &temp));
            element_size = static_cast<size_t>(temp);
          }
          element.reserve(element_size);
          for (size_t j = 0; j < element_size; ++j) {
            element.emplace_back();
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                full_name(strings::StrCat(kCache, "[", i, "][", j, "]")),
                &element.back()));
          }
          cache_->emplace_back(std::move(element));
        }
        if (reader->Contains(full_name(kCacheCompleted))) {
          cache_->Complete();
        }
      }
      InitializeIterator();
      TF_RETURN_IF_ERROR(iterator_->Initialize(ctx));
      return RestoreInput(ctx, reader, iterator_);
    }

   private:
    class MemoryWriterIterator : public DatasetIterator<MemoryDataset> {
     public:
      explicit MemoryWriterIterator(const Params& params, MemoryCache* cache)
          : DatasetIterator<MemoryDataset>(params), cache_(cache) {
        CHECK(cache_);
      }

      ~MemoryWriterIterator() override {
        mutex_lock l(mu_);
        if (cache_->size() > 0 && !cache_->IsCompleted()) {
          LOG(WARNING)
              << "The calling iterator did not fully read the dataset being "
                 "cached. In order to avoid unexpected truncation of the "
                 "dataset, the partially cached contents of the dataset "
                 "will be discarded. This can happen if you have an input "
                 "pipeline similar to `dataset.cache().take(k).repeat()`. "
                 "You should use `dataset.take(k).cache().repeat()` instead.";
          cache_->Reset();
        }
      }

      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
        if (*end_of_sequence) {
          cache_->Complete();
          return Status::OK();
        }
        RecordBufferEnqueue(ctx, *out_tensors);
        cache_->emplace_back(*out_tensors);
        return Status::OK();
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        return SaveInput(writer, input_impl_);
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        return RestoreInput(ctx, reader, input_impl_);
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      MemoryCache* const cache_ GUARDED_BY(mu_);  // not owned.
    };                                            // MemoryWriterIterator

    class MemoryReaderIterator : public DatasetIterator<MemoryDataset> {
     public:
      explicit MemoryReaderIterator(const Params& params, MemoryCache* cache)
          : DatasetIterator<MemoryDataset>(params), cache_(cache), index_(0) {
        CHECK(cache);
      }

      Status Initialize(IteratorContext* ctx) override {
        // The memory allocated for the cache is owned by the parent
        // dataset but performance modeling uses the iterator abstraction and
        // thus we record the memory allocated for the cache here. The caveat
        // is that this is incorrect if there are concurrent instances of this
        // iterator.
        tf_shared_lock l(mu_);
        for (size_t i = 0; i < cache_->size(); ++i) {
          RecordBufferEnqueue(ctx, cache_->at(i));
        }
        return Status::OK();
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (index_ < cache_->size()) {
          const std::vector<Tensor>& cache_tensors = cache_->at(index_);
          out_tensors->insert(out_tensors->begin(), cache_tensors.begin(),
                              cache_tensors.end());
          index_++;
          *end_of_sequence = false;
          return Status::OK();
        } else {
          *end_of_sequence = true;
          return Status::OK();
        }
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kIndex), index_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        {
          int64 temp;
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kIndex), &temp));
          index_ = static_cast<size_t>(temp);
        }
        return Status::OK();
      }

     private:
      mutex mu_;
      MemoryCache* const cache_ GUARDED_BY(mu_);  // not owned.
      size_t index_ GUARDED_BY(mu_);
    };  // MemoryReaderIterator

    void InitializeIterator() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      switch (mode_) {
        case Mode::read:
          iterator_ = absl::make_unique<MemoryReaderIterator>(
              MemoryReaderIterator::Params{dataset(),
                                           strings::StrCat(prefix(), kImpl)},
              cache_);
          break;
        case Mode::write:
          iterator_ = absl::make_unique<MemoryWriterIterator>(
              MemoryWriterIterator::Params{dataset(),
                                           strings::StrCat(prefix(), kImpl)},
              cache_);
      }
    }

    mutex mu_;
    MemoryCache* cache_ GUARDED_BY(mu_);  // not owned.
    enum Mode { read, write };
    Mode mode_ GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> iterator_ GUARDED_BY(mu_);
  };  // MemoryIterator

  const DatasetBase* const input_;
  MemoryCache* cache_ = nullptr;
};  // MemoryDataset

class CacheDatasetOp::MemoryDatasetV2 : public CacheDatasetOp::MemoryDataset {
 public:
  explicit MemoryDatasetV2(OpKernelContext* ctx, const DatasetBase* input,
                           MemoryCache* cache,
                           std::unique_ptr<OwnedResourceHandle> handle)
      : MemoryDataset(ctx, input, cache), handle_(std::move(handle)) {}

  Status CheckExternalState() const override {
    return errors::FailedPrecondition(DebugString(),
                                      " depends on memory cache resource.");
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));
    Node* filename_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(tstring(""), &filename_node));
    Node* resource_handle_node = nullptr;
    Tensor handle(DT_RESOURCE, TensorShape({}));
    handle.scalar<ResourceHandle>()() = handle_->handle();
    TF_RETURN_IF_ERROR(b->AddTensor(handle, &resource_handle_node));
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {input_node, filename_node, resource_handle_node}, output));
    return Status::OK();
  }

 private:
  std::unique_ptr<OwnedResourceHandle> handle_;
};

CacheDatasetOp::CacheDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx),
      op_version_(ctx->def().op() == kCacheDataset ? 1 : 2) {}

void CacheDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                 DatasetBase** output) {
  // Parse out the filenames tensor.
  tstring filename;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<tstring>(ctx, kFileName, &filename));

  if (filename.empty()) {
    if (op_version_ == 2) {
      MemoryCache* cache = nullptr;
      OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &cache));

      // Create a fresh handle for the resource because the input handle can
      // become invalid after this op executes.
      std::unique_ptr<OwnedResourceHandle> handle;
      OP_REQUIRES_OK(
          ctx, OwnedResourceHandle::Create(ctx, cache, kMemoryCache, &handle));

      // Ownership of cache is transferred onto `MemoryDatasetV2`.
      *output = new MemoryDatasetV2(ctx, input, cache, std::move(handle));
    } else {
      *output = new MemoryDataset(ctx, input, /*cache=*/nullptr);
    }
  } else {
    if (op_version_ == 2) {
      *output =
          new FileDatasetV2(ctx, input, filename, ctx->env(), ctx->input(2));
    } else {
      *output = new FileDataset(ctx, input, filename, ctx->env());
    }
  }
}

namespace {
REGISTER_KERNEL_BUILDER(Name("CacheDataset").Device(DEVICE_CPU),
                        CacheDatasetOp);
REGISTER_KERNEL_BUILDER(Name("CacheDatasetV2").Device(DEVICE_CPU),
                        CacheDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow