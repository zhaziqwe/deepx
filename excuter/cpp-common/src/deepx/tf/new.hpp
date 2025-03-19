class NewTensor : public TF
{
public:
    // ... 其他现有代码 ...

    shared_ptr<TF> clone() const override {
        return make_shared<NewTensor>(*this);
    }
}; 